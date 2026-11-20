from typing import Dict, Any, List, Optional
from app.graph.state import TaskResult, TaskStatus, ExecutionPlan
from app.core.logging import logger


class ResultAggregator:
    """
    ResultAggregator collects, merges, and formats outputs from all completed tasks.
    Preserves task IDs, deduplicates citation metadata, compiles study artifacts,
    and produces the final grounded answer.
    """

    def aggregate(
        self,
        task_results: List[TaskResult],
        plan: Optional[ExecutionPlan] = None,
        original_query: str = "",
    ) -> Dict[str, Any]:
        """
        Aggregates task outputs into a unified response structure.
        """
        text_outputs: List[str] = []
        doc_citations: List[Dict[str, Any]] = []
        paper_citations: List[Dict[str, Any]] = []
        study_artifacts: List[Dict[str, Any]] = []
        retrieved_chunks: List[Dict[str, Any]] = []
        research_sources: List[Dict[str, Any]] = []
        completed_workers = set()

        seen_chunk_ids = set()
        seen_paper_ids = set()

        for res in task_results:
            if res.status != TaskStatus.COMPLETED or not res.output:
                continue

            completed_workers.add(res.worker)

            # Extract citations
            if res.citations:
                for d in res.citations.get("documents", []):
                    cid = d.get("chunk_id") or f"{d.get('filename')}-{d.get('page_number')}"
                    if cid not in seen_chunk_ids:
                        seen_chunk_ids.add(cid)
                        doc_citations.append(d)

                for p in res.citations.get("research_papers", []):
                    pid = p.get("pmid") or p.get("doi") or p.get("title")
                    if pid not in seen_paper_ids:
                        seen_paper_ids.add(pid)
                        paper_citations.append(p)

            # Worker-specific output extraction
            if isinstance(res.output, dict):
                # RAG chunks
                if "chunks" in res.output and isinstance(res.output["chunks"], list):
                    for c in res.output["chunks"]:
                        cid = c.get("chunk_id") or f"{c.get('filename')}-{c.get('page_number')}"
                        if cid not in seen_chunk_ids:
                            retrieved_chunks.append(c)

                # Research articles
                if "articles" in res.output and isinstance(res.output["articles"], list):
                    for a in res.output["articles"]:
                        research_sources.append(a)

                # Study artifacts
                if "items" in res.output and res.worker in ("mcq", "flashcard"):
                    study_artifacts.extend(res.output["items"])

                # Text output
                text = res.output.get("text", "")
                if text:
                    # If this is a primary explanatory worker (tutor or direct_answer)
                    if res.worker in ("tutor", "direct_answer"):
                        text_outputs.append(text)
                    # For study workers, include their rendered cards
                    elif res.worker in ("mcq", "flashcard"):
                        text_outputs.append(text)
            elif isinstance(res.output, str):
                text_outputs.append(res.output)

        # Check if all tasks failed or produced no text
        if not text_outputs:
            failed_tasks = [r for r in task_results if r.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)]
            if failed_tasks:
                errors = "; ".join(f"{f.task_id} ({f.worker}): {f.error}" for f in failed_tasks)
                final_text = (
                    "I encountered an issue completing some of the requested study tasks:\n"
                    f"_{errors}_\n\n"
                    "Please try rephrasing your request or asking the questions individually."
                )
            else:
                final_text = "I have processed your query, but no substantive answer could be generated."
        else:
            final_text = "\n\n---\n\n".join(text_outputs)

        # Check evidence limitation notice
        has_retrieval_task = any(r.worker in ("rag", "research") for r in task_results)
        if has_retrieval_task and not doc_citations and not paper_citations:
            final_text = (
                "**Evidence Limitation Notice:**\n"
                "No relevant passages were found in your uploaded documents or peer-reviewed queries for this specific search.\n\n"
                + final_text
            )

        citations = {
            "documents": doc_citations,
            "research_papers": paper_citations,
        }

        return {
            "final_answer": final_text,
            "answer": final_text,  # Backward compatibility
            "citations": citations,
            "study_artifacts": study_artifacts,
            "retrieved_chunks": retrieved_chunks,
            "research_sources": research_sources,
        }
