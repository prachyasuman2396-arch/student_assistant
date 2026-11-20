import pytest
from unittest.mock import AsyncMock, MagicMock
from app.memory.service import MemoryService
from app.db.models.episodic_memory import EpisodicMemory
from app.db.models.memory import LongTermMemory
from app.db.models.profile import UserProfile
from app.llm.schemas import LLMResponse, ProviderName


@pytest.mark.asyncio
async def test_memory_service_trim_messages():
    svc = MemoryService()
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
    trimmed = svc.trim_messages(messages, max_messages=8)
    assert len(trimmed) == 8
    assert trimmed[0]["content"] == "msg 17"
    assert trimmed[-1]["content"] == "msg 24"


@pytest.mark.asyncio
async def test_memory_service_retrieve_memories_for_query():
    mock_p_repo = MagicMock()
    mock_m_repo = MagicMock()
    mock_e_repo = MagicMock()

    # Mock profile
    mock_p_obj = MagicMock()
    mock_p_obj.preferred_name = "Alice"
    mock_p_obj.academic_discipline = "Endodontics"
    mock_p_obj.current_level = "Postgraduate"
    mock_p_obj.explanation_style = "case_based"
    mock_p_obj.learning_goals = ["Board Certification"]
    mock_p_repo.get_by_user_id = AsyncMock(return_value=mock_p_obj)

    # Mock semantic memory
    mock_mem = MagicMock()
    mock_mem.id = "mem-1"
    mock_mem.topic = "Study Preferences"
    mock_mem.content = "Prefers clinical case analogies"
    mock_mem.memory_type = "preference"
    mock_mem.importance = 2
    mock_m_repo.search_memories = AsyncMock(return_value=[mock_mem])

    # Mock episodic memory
    mock_ep = MagicMock()
    mock_ep.id = "ep-1"
    mock_ep.event_type = "quiz_completed"
    mock_ep.topic = "Pulpitis"
    mock_ep.summary = "Scored 4/5 on pulpitis diagnosis quiz"
    mock_ep.importance = 2
    mock_e_repo.search_episodes = AsyncMock(return_value=[mock_ep])

    svc = MemoryService(
        profile_repo=mock_p_repo,
        memory_repo=mock_m_repo,
        episodic_repo=mock_e_repo,
    )

    bundle = await svc.retrieve_memories_for_query(user_id="user-1", query="pulpitis")
    assert bundle["profile"]["preferred_name"] == "Alice"
    assert len(bundle["semantic_memories"]) == 1
    assert bundle["semantic_memories"][0]["content"] == "Prefers clinical case analogies"
    assert len(bundle["episodic_memories"]) == 1
    assert "Scored 4/5" in bundle["episodic_memories"][0]["summary"]


@pytest.mark.asyncio
async def test_memory_service_extract_and_update_with_anti_hallucination(mock_router):
    mock_m_repo = MagicMock()
    mock_e_repo = MagicMock()

    mock_m_repo.add_memory = AsyncMock(
        return_value=MagicMock(id="mem-new", topic="Endodontics", content="Goal is to pass BDS Part 2", memory_type="study_goal")
    )
    mock_e_repo.add_episode = AsyncMock(
        return_value=MagicMock(id="ep-new", topic="Endodontics", summary="Completed 5 MCQs", event_type="revision_milestone")
    )

    mock_router.execute_with_fallback = AsyncMock(
        return_value=LLMResponse(
            content="""{
                "semantic_memories": [
                    {
                        "memory_type": "study_goal",
                        "topic": "Endodontics",
                        "content": "Goal is to pass BDS Part 2",
                        "importance": 2
                    }
                ],
                "episodic_memories": []
            }""",
            provider=ProviderName.GROQ,
            model="llama-3.1-8b-instant",
        )
    )

    svc = MemoryService(
        router=mock_router,
        memory_repo=mock_m_repo,
        episodic_repo=mock_e_repo,
    )

    # Student explicitly states goal
    res = await svc.extract_and_update_memories(
        user_id="user-1",
        query="Please remember this: my goal is to pass BDS Part 2 in Endodontics.",
        final_answer="Noted your goal!",
    )
    assert len(res["semantic"]) == 1
    assert res["semantic"][0]["content"] == "Goal is to pass BDS Part 2"
    mock_m_repo.add_memory.assert_called_once()


@pytest.mark.asyncio
async def test_memory_service_ignores_routine_factual_questions(mock_router):
    mock_m_repo = MagicMock()
    mock_m_repo.add_memory = AsyncMock()

    svc = MemoryService(
        router=mock_router,
        memory_repo=mock_m_repo,
    )

    # Simple factual question without explicit memory trigger
    res = await svc.extract_and_update_memories(
        user_id="user-1",
        query="What are the histologic zones of the dental pulp?",
        final_answer="The zones include odontoblastic layer, cell-free zone of Weil, cell-rich zone...",
    )
    # Should not trigger extraction or persistence
    assert len(res["semantic"]) == 0
    mock_m_repo.add_memory.assert_not_called()


@pytest.mark.asyncio
async def test_memory_service_tenant_isolation_delete():
    mock_m_repo = MagicMock()
    mock_m_repo.delete_memory = AsyncMock(return_value=True)

    svc = MemoryService(memory_repo=mock_m_repo)
    success = await svc.delete_memory(memory_id="mem-1", user_id="user-1", memory_type="semantic")
    assert success is True
    mock_m_repo.delete_memory.assert_called_once_with("mem-1", "user-1")
