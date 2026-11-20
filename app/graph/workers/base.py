import abc
from typing import Dict, Any, Optional
from app.graph.state import Task, TaskResult, TaskStatus


class BaseWorker(abc.ABC):
    """
    Abstract contract for all specialized multi-agent workers.
    Workers receive an input task and read-only context of completed upstream tasks,
    and must return a structured TaskResult. Workers never directly mutate global state
    or arbitrarily dispatch other workers.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Worker name identifying this worker in the registry."""
        pass

    @abc.abstractmethod
    async def execute(
        self,
        task: Task,
        context: Dict[str, Any],
    ) -> TaskResult:
        """
        Execute the task.
        Args:
            task: The Task specification to execute.
            context: Execution context containing:
                - user_id: str
                - conversation_id: str
                - upstream_results: Dict[str, TaskResult] (results of tasks this task depends on)
                - profile: Optional[Dict[str, Any]]
                - semantic_memories: List[Dict[str, Any]]
                - episodic_memories: List[Dict[str, Any]]
                - document_id: Optional[str]
        Returns:
            TaskResult: Standardized output, status, citations, error, and latency.
        """
        pass
