from enum import Enum
from typing import Annotated, Dict, Any, List, Optional, Union
import operator
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Task(BaseModel):
    task_id: str = Field(description="Unique identifier for the task, e.g., T1, T2")
    worker: str = Field(description="Worker type: 'rag', 'research', 'tutor', 'mcq', 'flashcard', 'direct_answer'")
    description: str = Field(description="Clear description of what this task must accomplish")
    dependencies: List[str] = Field(default_factory=list, description="List of task_ids that must finish before this task runs")
    expected_output: str = Field(description="Description of the expected artifact or result from this task")
    input_mapping: Dict[str, Any] = Field(default_factory=dict, description="Parameters or upstream data keys needed by worker")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current execution status of this task")
    retry_count: int = Field(default=0, description="Number of times this task has been retried")
    max_retries: int = Field(default=2, description="Maximum retry attempts allowed for this task")


class ExecutionPlan(BaseModel):
    objective: str = Field(description="Overall goal of the execution plan")
    tasks: List[Task] = Field(default_factory=list, description="Ordered or dependency-linked list of tasks")
    final_response_strategy: str = Field(
        default="synthesize",
        description="Strategy for aggregating results: 'synthesize', 'direct', or 'composite'"
    )
    is_direct: bool = Field(
        default=False,
        description="True if query is simple and does not require complex multi-step decomposition"
    )


class TaskResult(BaseModel):
    task_id: str
    worker: str
    status: TaskStatus
    output: Any = None
    citations: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0


def merge_task_results(
    existing: Optional[List[Dict[str, Any]]],
    new_results: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Reducer function for LangGraph to merge worker results deterministically by task_id.
    Prevents duplicate entries when parallel tasks complete.
    """
    results_map: Dict[str, Dict[str, Any]] = {}
    for r in (existing or []):
        if isinstance(r, dict) and "task_id" in r:
            results_map[r["task_id"]] = r
    for r in (new_results or []):
        if isinstance(r, dict) and "task_id" in r:
            results_map[r["task_id"]] = r
    return list(results_map.values())


class AgentState(TypedDict, total=False):
    """
    Typed workflow state for the Planner–Orchestrator–Multi-Agent Architecture.
    Maintains 100% backward compatibility with legacy fields while introducing
    orchestration and memory channels with LangGraph reducers.
    """
    # Identifiers & Context
    messages: List[Dict[str, Any]]
    user_id: str
    conversation_id: str
    document_id: Optional[str]
    academic_level: Optional[str]

    # Memory Subsystems
    profile: Optional[Dict[str, Any]]
    long_term_memories: List[Dict[str, Any]]
    semantic_memories: List[Dict[str, Any]]
    episodic_memories: List[Dict[str, Any]]
    extracted_memories: List[Dict[str, Any]]

    # Planner & Validation
    plan: Optional[Dict[str, Any]]
    plan_is_valid: bool
    plan_validation_error: Optional[str]
    plan_revision_count: int

    # Orchestration Channels (with reducers for concurrent worker dispatch)
    task_results: Annotated[List[Dict[str, Any]], merge_task_results]
    completed_task_ids: Annotated[List[str], operator.add]
    failed_task_ids: Annotated[List[str], operator.add]
    active_tasks: List[str]

    # Legacy Classification & Retrieval Channels (Preserved for compatibility)
    intent: str
    task_complexity: str
    search_query: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    research_sources: List[Dict[str, Any]]
    evidence_sufficient: bool
    retry_count: int

    # Synthesis & Final Output
    answer: str
    final_answer: str
    citations: Dict[str, Any]
    study_artifacts: List[Dict[str, Any]]

    # Telemetry
    selected_provider: Optional[str]
    selected_model: Optional[str]
    error: Optional[str]


# Type alias for multi-agent state
MultiAgentState = AgentState
