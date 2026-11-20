import pytest
from unittest.mock import AsyncMock, MagicMock
from app.graph.state import AgentState, ExecutionPlan, Task, TaskStatus
from app.graph.nodes import DentalGraphNodes
from app.graph.workflow import create_study_graph
from app.llm.schemas import LLMResponse, ProviderName
from app.rag.vector_store import RetrievedChunk
from app.research.schemas import ResearchArticle, EvidenceLevel


@pytest.mark.asyncio
async def test_end_to_end_combined_rag_research_and_mcq_workflow(mock_router):
    """
    Tests complete Planner -> Plan Validator -> Orchestrator -> Workers -> Aggregator -> Memory Update
    pipeline for a compound query:
    'Explain irreversible pulpitis using my uploaded textbook, compare with recent research papers and generate five MCQs.'
    """
    # 1. Setup Mock Router to return structured execution plan on planning,
    # and synthesized text on tutor synthesis
    async def mock_execute(*args, **kwargs):
        task_type = kwargs.get("task_type")
        prompt_content = str(kwargs.get("messages", []))

        if "You are an expert AI Study Architect" in prompt_content:
            return LLMResponse(
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
                            "input_mapping": {"query": "irreversible pulpitis"}
                        },
                        {
                            "task_id": "T2",
                            "worker": "research",
                            "description": "Retrieve recent PubMed clinical research",
                            "dependencies": [],
                            "expected_output": "Research papers",
                            "input_mapping": {"query": "irreversible pulpitis"}
                        },
                        {
                            "task_id": "T3",
                            "worker": "tutor",
                            "description": "Synthesize comparative analysis",
                            "dependencies": ["T1", "T2"],
                            "expected_output": "Grounded comparative synthesis",
                            "input_mapping": {"query": "irreversible pulpitis"}
                        },
                        {
                            "task_id": "T4",
                            "worker": "mcq",
                            "description": "Generate practice examination MCQs",
                            "dependencies": ["T3"],
                            "expected_output": "MCQ questions",
                            "input_mapping": {"count": 3}
                        }
                    ]
                }""",
                provider=ProviderName.GROQ,
                model="llama-3.3-70b-versatile",
            )
        elif task_type == "research_synthesis" or "synthesize" in prompt_content.lower():
            return LLMResponse(
                content="Irreversible pulpitis exhibits spontaneous, lingering pain. Textbook emphasizes microabscess formation while recent PubMed trials highlight vital pulp therapy efficacy.",
                provider=ProviderName.GROQ,
                model="llama-3.3-70b-versatile",
            )
        elif task_type == "study_generation" or "mcq" in prompt_content.lower():
            return LLMResponse(
                content="""[
                    {
                        "question": "Which clinical symptom is hallmark of irreversible pulpitis?",
                        "options": ["Lingering thermal pain", "Transient sweet sensitivity", "Mobility", "Fistula"],
                        "correct_answer": "Lingering thermal pain",
                        "explanation": "Lingering pain to heat or cold lasting more than 30 seconds indicates irreversible pulpitis.",
                        "difficulty": "medium"
                    }
                ]""",
                provider=ProviderName.GROQ,
                model="llama-3.3-70b-versatile",
            )
        # Default response
        return LLMResponse(
            content="Standard pedagogical explanation.",
            provider=ProviderName.GROQ,
            model="llama-3.3-70b-versatile",
        )

    mock_router.execute_with_fallback = AsyncMock(side_effect=mock_execute)

    # 2. Mock RAG retriever
    mock_retriever = MagicMock()
    mock_chunk = RetrievedChunk(
        chunk_id="chk-1",
        document_id="doc-101",
        filename="Ingle_Endodontics.pdf",
        page_number=45,
        content="Histopathologically, irreversible pulpitis demonstrates localized microabscesses with PMNs.",
        similarity_score=0.88,
    )
    mock_retriever.retrieve = AsyncMock(return_value=[mock_chunk])

    # 3. Mock Research service
    mock_research = MagicMock()
    mock_article = ResearchArticle(
        title="Modern Outcomes of Vital Pulp Therapy in Irreversible Pulpitis",
        authors="Taha, N., Al-Khatib, M.",
        journal="J Endod",
        publication_date="2023",
        pmid="36541234",
        doi="10.1016/j.joen.2023.01.002",
        evidence_level=EvidenceLevel.CLINICAL_GUIDELINE,
        abstract="Vital pulp therapy using bioceramics demonstrated 92% success at 2-year follow-up.",
    )
    mock_research.search_literature = AsyncMock(return_value=[mock_article])

    # 4. Construct Nodes & Graph
    nodes = DentalGraphNodes(
        router=mock_router,
        research_service=mock_research,
        retriever_factory=lambda: mock_retriever,
    )

    graph = create_study_graph(nodes)

    initial_state: AgentState = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Explain irreversible pulpitis using my uploaded textbook, "
                    "compare it with recent research papers and generate five MCQs."
                ),
            }
        ],
        "user_id": "student-test-1",
        "conversation_id": "conv-test-multiagent-1",
        "document_id": "doc-101",
    }

    final_state = await graph.ainvoke(initial_state)

    # Assertions on final aggregated state
    assert final_state["final_answer"] != ""
    assert "Irreversible pulpitis" in final_state["final_answer"]
    assert "Lingering thermal pain" in final_state["final_answer"]

    # Citations verification
    citations = final_state.get("citations", {})
    assert len(citations.get("documents", [])) >= 1
    assert citations["documents"][0]["filename"] == "Ingle_Endodontics.pdf"
    assert len(citations.get("research_papers", [])) >= 1
    assert citations["research_papers"][0]["pmid"] == "36541234"

    # Tasks verification in state
    task_results = final_state.get("task_results", [])
    assert len(task_results) == 4
    worker_names = {r["worker"] for r in task_results}
    assert worker_names == {"rag", "research", "tutor", "mcq"}
    assert all(r["status"] == TaskStatus.COMPLETED for r in task_results)


@pytest.mark.asyncio
async def test_end_to_end_flashcard_generation_workflow(mock_router):
    """
    Tests direct flashcard generation task flow.
    """
    async def mock_execute(*args, **kwargs):
        prompt_content = str(kwargs.get("messages", []))
        if "You are an expert AI Study Architect" in prompt_content:
            return LLMResponse(
                content="""{
                    "objective": "Generate active recall flashcards on local anesthetics",
                    "is_direct": false,
                    "final_response_strategy": "direct",
                    "tasks": [
                        {
                            "task_id": "T1",
                            "worker": "flashcard",
                            "description": "Generate 3 flashcards on lidocaine and articaine",
                            "dependencies": [],
                            "expected_output": "Flashcard items",
                            "input_mapping": {"count": 3}
                        }
                    ]
                }""",
                provider=ProviderName.GROQ,
                model="llama-3.3-70b-versatile",
            )
        return LLMResponse(
            content="""[
                {
                    "question": "What is the maximum recommended dose of 2% Lidocaine with 1:100,000 epinephrine?",
                    "answer": "4.4 mg/kg (up to an absolute maximum of 300 mg).",
                    "difficulty": "medium"
                }
            ]""",
            provider=ProviderName.GROQ,
            model="llama-3.3-70b-versatile",
        )

    mock_router.execute_with_fallback = AsyncMock(side_effect=mock_execute)
    nodes = DentalGraphNodes(router=mock_router)
    graph = create_study_graph(nodes)

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": "Generate study flashcards on local anesthetics."}],
        "user_id": "student-2",
        "conversation_id": "conv-flashcards",
    }

    final_state = await graph.ainvoke(initial_state)
    assert "Study Flashcards" in final_state["final_answer"]
    assert "4.4 mg/kg" in final_state["final_answer"]
    assert len(final_state.get("study_artifacts", [])) >= 1
