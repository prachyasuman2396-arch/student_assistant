from typing import Literal
from app.graph.state import AgentState

MAX_PLAN_REVISIONS = 2


def route_after_plan_validation(
    state: AgentState,
) -> Literal["orchestrator", "planner"]:
    """
    Routes after plan validation:
    - If plan is valid: proceeds to orchestrator.
    - If plan is invalid and revision attempts < MAX_PLAN_REVISIONS: loops back to planner with feedback.
    - If plan is invalid and revisions exhausted: proceeds to orchestrator which applies safe fallback.
    """
    is_valid = state.get("plan_is_valid", False)
    revisions = state.get("plan_revision_count", 0)

    if is_valid:
        return "orchestrator"

    if revisions < MAX_PLAN_REVISIONS:
        return "planner"

    return "orchestrator"


# Legacy routing preserved for backwards compatibility with tests
def route_by_intent(
    state: AgentState,
) -> Literal["tutor", "web_research", "document_retrieval", "hybrid_retrieval", "study_tools"]:
    intent = state.get("intent", "tutor")
    if intent in ("tutor", "web_research", "document_retrieval", "hybrid_retrieval", "study_tools"):
        return intent
    return "tutor"


def route_after_verification(
    state: AgentState,
) -> Literal["save_turn", "web_research"]:
    evidence_sufficient = state.get("evidence_sufficient", True)
    retry_count = state.get("retry_count", 0)
    intent = state.get("intent", "tutor")

    if not evidence_sufficient and retry_count < 1 and intent in ("document_retrieval", "web_research"):
        return "web_research"

    return "save_turn"
