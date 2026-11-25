import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from app.core.observability import configure_observability, get_langsmith_client


def score_faithfulness(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    low = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in low)
    return round(hits / len(expected_keywords), 4)


def score_citation_precision(answer: str) -> float:
    citations = re.findall(r"\[[^\]]+,\s*p\.\s*\d+[^\]]*\]", answer)
    return 1.0 if citations else 0.0


def score_answer_relevancy(question: str, answer: str) -> float:
    q_terms = {t.lower() for t in re.findall(r"\w+", question) if len(t) > 3}
    if not q_terms:
        return 1.0
    a_low = answer.lower()
    overlap = sum(1 for t in q_terms if t in a_low)
    return round(min(1.0, overlap / max(1, len(q_terms) * 0.4)), 4)


async def run_evaluation(dataset_path: str, offline_mock: bool = True) -> Dict[str, Any]:
    configure_observability()
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    ls_client = get_langsmith_client()
    results: List[Dict[str, Any]] = []

    for item in items:
        q = item["question"]
        kws = item.get("expected_answer_keywords", [])
        if offline_mock:
            # Deterministic grounded synthetic response for CI verification
            answer = (
                f"Based on retrieved textbook evidence [Endodontics Textbook, p.142, text]: "
                + ", ".join(kws)
                + f". This directly addresses: {q}"
            )
        else:
            from app.llm.router import LLMRouter
            from app.llm.schemas import TaskType, ComplexityLevel, LLMMessage
            router = LLMRouter()
            resp = await router.execute_with_fallback(
                task_type=TaskType.TUTOR,
                messages=[LLMMessage(role="user", content=q)],
                complexity=ComplexityLevel.MEDIUM,
            )
            answer = resp.content

        faith = score_faithfulness(answer, kws)
        cit_prec = score_citation_precision(answer)
        rel = score_answer_relevancy(q, answer)

        results.append({
            "id": item["id"],
            "faithfulness": faith,
            "citation_precision": cit_prec,
            "answer_relevancy": rel,
        })

    n = max(1, len(results))
    summary = {
        "total_items": len(results),
        "langsmith_connected": ls_client is not None,
        "mean_faithfulness": round(sum(r["faithfulness"] for r in results) / n, 4),
        "mean_citation_precision": round(sum(r["citation_precision"] for r in results) / n, 4),
        "mean_answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / n, 4),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSmith RAG evaluation harness")
    parser.add_argument("--dataset", default="evaluation/dental_eval_dataset.json")
    parser.add_argument("--live", action="store_true", help="Invoke live LLM router")
    args = parser.parse_args()
    asyncio.run(run_evaluation(args.dataset, offline_mock=not args.live))
