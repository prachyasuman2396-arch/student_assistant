import pytest
from unittest.mock import AsyncMock, MagicMock
from app.graph.planner import PlannerAgent
from app.graph.state import ExecutionPlan, Task, TaskStatus
from app.llm.schemas import LLMResponse, ProviderName


@pytest.mark.asyncio
async def test_planner_direct_path_for_simple_greeting(mock_router):
    planner = PlannerAgent(router=mock_router)
    plan = await planner.generate_plan(query="Hello there!")
    assert plan.is_direct is True
    assert len(plan.tasks) == 1
    assert plan.tasks[0].worker == "direct_answer"
    assert plan.tasks[0].dependencies == []


@pytest.mark.asyncio
async def test_planner_direct_path_for_simple_factual_question(mock_router):
    mock_router.execute_with_fallback = AsyncMock(
        return_value=LLMResponse(
            content="""{
                "objective": "Explain enamel composition",
                "is_direct": true,
                "final_response_strategy": "direct",
                "tasks": [
                    {
                        "task_id": "T1",
                        "worker": "tutor",
                        "description": "Provide pedagogical explanation of enamel",
                        "dependencies": [],
                        "expected_output": "Enamel explanation",
                        "input_mapping": {"query": "What is enamel made of?"}
                    }
                ]
            }""",
            provider=ProviderName.GROQ,
            model="llama-3.3-70b-versatile",
        )
    )
    planner = PlannerAgent(router=mock_router)
    plan = await planner.generate_plan(query="What is enamel made of?")
    assert plan.is_direct is True
    assert len(plan.tasks) == 1
    assert plan.tasks[0].worker == "tutor"
    assert plan.tasks[0].dependencies == []


@pytest.mark.asyncio
async def test_planner_complex_multi_task_decomposition(mock_router):
    mock_router.execute_with_fallback = AsyncMock(
        return_value=LLMResponse(
            content="""{
                "objective": "Compare textbook and PubMed research on irreversible pulpitis and generate MCQs",
                "is_direct": false,
                "final_response_strategy": "synthesize",
                "tasks": [
                    {
                        "task_id": "T1",
                        "worker": "rag",
                        "description": "Retrieve textbook excerpts on irreversible pulpitis",
                        "dependencies": [],
                        "expected_output": "Textbook passages",
                        "input_mapping": {"query": "irreversible pulpitis histology diagnosis"}
                    },
                    {
                        "task_id": "T2",
                        "worker": "research",
                        "description": "Retrieve recent clinical trials and research papers",
                        "dependencies": [],
                        "expected_output": "Research articles",
                        "input_mapping": {"query": "irreversible pulpitis treatment outcome"}
                    },
                    {
                        "task_id": "T3",
                        "worker": "tutor",
                        "description": "Compare textbook concepts with modern research findings",
                        "dependencies": ["T1", "T2"],
                        "expected_output": "Comprehensive synthesis",
                        "input_mapping": {"query": "compare textbook and research"}
                    },
                    {
                        "task_id": "T4",
                        "worker": "mcq",
                        "description": "Generate 5 clinical MCQs",
                        "dependencies": ["T3"],
                        "expected_output": "5 MCQs",
                        "input_mapping": {"count": 5}
                    }
                ]
            }""",
            provider=ProviderName.GROQ,
            model="llama-3.3-70b-versatile",
        )
    )
    planner = PlannerAgent(router=mock_router)
    plan = await planner.generate_plan(
        query="Explain irreversible pulpitis using my uploaded textbook, compare it with recent research papers and generate five MCQs.",
        document_id="doc-456",
    )
    assert plan.is_direct is False
    assert len(plan.tasks) == 4
    # Check concurrent tasks T1 and T2
    assert plan.tasks[0].dependencies == []
    assert plan.tasks[1].dependencies == []
    # Check dependent task T3
    assert set(plan.tasks[2].dependencies) == {"T1", "T2"}
    # Check downstream task T4
    assert plan.tasks[3].dependencies == ["T3"]


@pytest.mark.asyncio
async def test_planner_heuristic_fallback_when_llm_json_fails(mock_router):
    mock_router.execute_with_fallback = AsyncMock(
        return_value=LLMResponse(
            content="Invalid non-json response from model",
            provider=ProviderName.GROQ,
            model="llama-3.3-70b-versatile",
        )
    )
    planner = PlannerAgent(router=mock_router)
    plan = await planner.generate_plan(
        query="Compare my textbook with recent research papers on caries and create MCQs",
        document_id="doc-999",
    )
    assert plan is not None
    assert len(plan.tasks) >= 3
    workers = [t.worker for t in plan.tasks]
    assert "rag" in workers
    assert "research" in workers
    assert "tutor" in workers
