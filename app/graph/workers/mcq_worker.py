import json
import re
import time
from typing import Dict, Any, Optional, Callable, List
from app.graph.workers.base import BaseWorker
from app.graph.state import Task, TaskResult, TaskStatus
from app.llm.router import LLMRouter
from app.llm.schemas import TaskType, ComplexityLevel, LLMMessage
from app.prompts.study import QUIZ_GENERATION_PROMPT
from app.core.logging import logger


class MCQWorker(BaseWorker):
    """
    Specialized worker for generating structured examination MCQs with
    options, correct answers, and clinical/scientific explanations.
    """

    def __init__(self, router: LLMRouter, study_service_factory: Optional[Callable[[], Any]] = None):
        self.router = router
        self.study_service_factory = study_service_factory

    @property
    def name(self) -> str:
        return "mcq"

    async def execute(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        start_time = time.perf_counter()
        query = task.input_mapping.get("query") or context.get("query") or task.description
        upstream_results = context.get("upstream_results", {})
        num_questions = task.input_mapping.get("count", 5)
        user_id = context.get("user_id", "")

        # Incorporate upstream synthesized evidence if present
        source_context = ""
        for dep_id in task.dependencies:
            dep_res = upstream_results.get(dep_id)
            if dep_res and dep_res.output and isinstance(dep_res.output, dict):
                text = dep_res.output.get("text", "")
                if text:
                    source_context += f"\nContext from Task {dep_id}:\n{text}\n"

        prompt_text = (
            f"Generate {num_questions} high-yield examination MCQs based on the following topic/material: '{query}'.\n"
            + (f"\nReference Material:\n{source_context}\n" if source_context else "")
        )

        messages = [
            LLMMessage(role="system", content=QUIZ_GENERATION_PROMPT),
            LLMMessage(role="user", content=prompt_text),
        ]

        try:
            res = await self.router.execute_with_fallback(
                task_type=TaskType.STUDY_GENERATION,
                messages=messages,
                complexity=ComplexityLevel.LOW,
                temperature=0.3,
            )

            raw = res.content.strip()
            clean_json = re.sub(r"^```json\s*", "", raw)
            clean_json = re.sub(r"\s*```$", "", clean_json)

            try:
                items = json.loads(clean_json)
                if not isinstance(items, list):
                    items = [items]
            except Exception:
                items = [{"question": "Review questions", "options": [], "explanation": res.content}]

            # Format human-readable output
            formatted_lines = [f"### Practice MCQs ({len(items)} Questions)\n"]
            for idx, item in enumerate(items, start=1):
                formatted_lines.append(f"**Q{idx}: {item.get('question')}**")
                opts = item.get("options", [])
                for opt in opts:
                    formatted_lines.append(f"- {opt}")
                if item.get("correct_answer"):
                    formatted_lines.append(f"*Correct Answer:* **{item.get('correct_answer')}**")
                if item.get("explanation"):
                    formatted_lines.append(f"*Explanation:* {item.get('explanation')}\n")

            rendered_text = "\n".join(formatted_lines)

            # Persist to study repository if service factory available
            if self.study_service_factory and user_id:
                try:
                    svc = self.study_service_factory()
                    for it in items:
                        if hasattr(svc, "repo") and hasattr(svc.repo, "create_card"):
                            await svc.repo.create_card(
                                user_id=user_id,
                                topic=query[:100],
                                card_type="mcq",
                                question=it.get("question", ""),
                                answer=it.get("correct_answer", ""),
                                explanation=it.get("explanation", ""),
                                options=it.get("options", []),
                                difficulty=it.get("difficulty", "medium"),
                            )
                except Exception as pe:
                    logger.warning(f"Could not persist MCQs in MCQWorker: {pe}")

            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.COMPLETED,
                output={"items": items, "text": rendered_text},
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        except Exception as e:
            logger.error(f"MCQWorker task '{task.task_id}' failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.FAILED,
                output=None,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
