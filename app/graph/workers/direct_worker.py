import time
from typing import Dict, Any
from app.graph.workers.base import BaseWorker
from app.graph.state import Task, TaskResult, TaskStatus
from app.llm.router import LLMRouter
from app.llm.schemas import TaskType, ComplexityLevel, LLMMessage


class DirectAnswerWorker(BaseWorker):
    """
    Lightweight fast worker for greetings and trivial single-turn conversational queries.
    """

    def __init__(self, router: LLMRouter):
        self.router = router

    @property
    def name(self) -> str:
        return "direct_answer"

    async def execute(self, task: Task, context: Dict[str, Any]) -> TaskResult:
        start_time = time.perf_counter()
        query = task.input_mapping.get("query") or context.get("query") or task.description

        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a helpful, encouraging student academic AI assistant. "
                    "Respond concisely and helpfully to the student's greeting or simple question."
                ),
            ),
            LLMMessage(role="user", content=query),
        ]

        try:
            res = await self.router.execute_with_fallback(
                task_type=TaskType.CONVERSATION,
                messages=messages,
                complexity=ComplexityLevel.LOW,
                temperature=0.6,
                max_tokens=250,
            )
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.COMPLETED,
                output={"text": res.content, "provider": res.provider.value, "model": res.model},
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                worker=self.name,
                status=TaskStatus.FAILED,
                output=None,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
            )
