from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.graph.state import AgentState
from app.graph.nodes import DentalGraphNodes
from app.graph.routing import route_after_plan_validation, route_by_intent


def create_planner_orchestrator_graph(nodes: DentalGraphNodes, checkpointer: BaseCheckpointSaver = None):
    """
    Constructs and compiles the upgraded Planner–Orchestrator–Multi-Agent LangGraph workflow.
    
    Architecture:
    START -> memory_retrieval -> planner -> plan_validator
    plan_validator -> [valid -> orchestrator | invalid (revisions < 2) -> planner | invalid (exhausted) -> orchestrator]
    orchestrator -> result_aggregator -> memory_update -> END
    """
    graph = StateGraph(AgentState)

    # 1. Register Multi-Agent Pipeline Nodes
    graph.add_node("memory_retrieval", nodes.retrieve_memories)
    graph.add_node("planner", nodes.plan_query)
    graph.add_node("plan_validator", nodes.validate_plan)
    graph.add_node("orchestrator", nodes.orchestrate_tasks)
    graph.add_node("result_aggregator", nodes.aggregate_results)
    graph.add_node("memory_update", nodes.update_memories)

    # 2. Sequential & Conditional Edges
    graph.add_edge(START, "memory_retrieval")
    graph.add_edge("memory_retrieval", "planner")
    graph.add_edge("planner", "plan_validator")

    # Bounded revision loop or proceed to orchestrator
    graph.add_conditional_edges(
        "plan_validator",
        route_after_plan_validation,
        {
            "orchestrator": "orchestrator",
            "planner": "planner",
        },
    )

    graph.add_edge("orchestrator", "result_aggregator")
    graph.add_edge("result_aggregator", "memory_update")
    graph.add_edge("memory_update", END)

    return graph.compile(checkpointer=checkpointer)


def create_legacy_study_graph(nodes: DentalGraphNodes, checkpointer: BaseCheckpointSaver = None):
    """
    Constructs the legacy linear state graph for full backwards compatibility.
    """
    graph = StateGraph(AgentState)

    graph.add_node("load_context", nodes.load_context)
    graph.add_node("classify_query", nodes.classify_query)
    graph.add_node("tutor", nodes.tutor)
    graph.add_node("document_retrieval", nodes.document_retrieval)
    graph.add_node("web_research", nodes.web_research)
    graph.add_node("hybrid_retrieval", nodes.hybrid_retrieval)
    graph.add_node("study_tools", nodes.study_tools)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("verify_answer", nodes.verify_answer)
    graph.add_node("save_turn", nodes.save_turn)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_query")

    graph.add_conditional_edges(
        "classify_query",
        route_by_intent,
        {
            "tutor": "tutor",
            "document_retrieval": "document_retrieval",
            "web_research": "web_research",
            "hybrid_retrieval": "hybrid_retrieval",
            "study_tools": "study_tools",
        },
    )

    graph.add_edge("document_retrieval", "synthesize")
    graph.add_edge("web_research", "synthesize")
    graph.add_edge("hybrid_retrieval", "synthesize")

    graph.add_edge("tutor", "verify_answer")
    graph.add_edge("study_tools", "verify_answer")
    graph.add_edge("synthesize", "verify_answer")

    graph.add_edge("verify_answer", "save_turn")
    graph.add_edge("save_turn", END)

    return graph.compile(checkpointer=checkpointer)


# Default factory: Upgraded Planner–Orchestrator graph
create_study_graph = create_planner_orchestrator_graph
create_dental_graph = create_study_graph
