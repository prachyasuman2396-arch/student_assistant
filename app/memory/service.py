import json
import re
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.repositories.profile_repo import ProfileRepository
from app.db.repositories.memory_repo import MemoryRepository
from app.db.repositories.episodic_repo import EpisodicRepository
from app.llm.router import LLMRouter
from app.llm.schemas import TaskType, ComplexityLevel, LLMMessage
from app.core.logging import logger


MEMORY_EXTRACTION_PROMPT = """You are an expert Educational Memory Auditor for a student learning system.
Analyze the student's latest message and the system's generated response to determine if any PERMANENT semantic study preferences or EPISODIC learning milestones should be recorded.

RULES:
1. NEVER record factual questions (e.g., "What is pulpitis?") as personal facts about the student.
2. Only record SEMANTIC facts if the student EXPLICITLY states a preference, study goal, or learning constraint.
   Examples of SEMANTIC facts:
   - "My goal is to master endodontics for the upcoming board exam" -> type="study_goal"
   - "Explain everything using bullet points and clinical case analogies" -> type="preference"
   - "Remember to keep track of local anesthesia dosages for my revision" -> type="revision_topic"
3. Only record EPISODIC events if a notable learning event occurred:
   - Student completed a quiz with a score or struggled with a specific concept.
   - Student completed a major multi-step revision session.
4. If nothing qualifies for persistence, return empty lists.

Output strictly as a JSON object:
{
  "semantic_memories": [
    {
      "memory_type": "preference" | "study_goal" | "saved_topic" | "revision_topic",
      "topic": "Brief topic",
      "content": "Specific durable preference or goal statement",
      "importance": 1 to 3
    }
  ],
  "episodic_memories": [
    {
      "event_type": "quiz_completed" | "struggled_concept" | "mastered_concept" | "revision_milestone",
      "topic": "Topic of the episode",
      "summary": "Concise factual summary of what happened during this learning session",
      "importance": 1 to 3
    }
  ]
}
"""


class MemoryService:
    """
    Unified memory management service.
    Orchestrates:
    - Short-term conversation history windowing and trimming
    - Semantic memory retrieval and deduplicated persistence
    - Episodic memory retrieval and structured event logging
    - Tenant isolation: All queries strictly scoped by user_id
    """

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        router: Optional[LLMRouter] = None,
        profile_repo: Optional[ProfileRepository] = None,
        memory_repo: Optional[MemoryRepository] = None,
        episodic_repo: Optional[EpisodicRepository] = None,
    ):
        self.session = session
        self.router = router or LLMRouter()
        self.profile_repo = profile_repo or (ProfileRepository(session) if session else None)
        self.memory_repo = memory_repo or (MemoryRepository(session) if session else None)
        self.episodic_repo = episodic_repo or (EpisodicRepository(session) if session else None)

    def trim_messages(
        self,
        messages: List[Dict[str, Any]],
        max_messages: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Sliding-window context management: preserves recent conversation turns
        to prevent unbounded message growth in state checkpoints.
        """
        if not messages:
            return []
        if len(messages) <= max_messages:
            return list(messages)
        # Keep the most recent max_messages turns
        return list(messages[-max_messages:])

    async def retrieve_memories_for_query(
        self,
        user_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Pre-planning memory retrieval: loads user profile, semantic preferences/goals,
        and relevant past learning episodes.
        """
        profile_data: Optional[Dict[str, Any]] = None
        semantic_memories: List[Dict[str, Any]] = []
        episodic_memories: List[Dict[str, Any]] = []

        if not user_id:
            return {
                "profile": None,
                "semantic_memories": [],
                "episodic_memories": [],
            }

        # 1. Load User Profile
        if self.profile_repo:
            try:
                p_obj = await self.profile_repo.get_by_user_id(user_id)
                if p_obj:
                    profile_data = {
                        "preferred_name": p_obj.preferred_name,
                        "academic_discipline": p_obj.academic_discipline,
                        "current_level": p_obj.current_level,
                        "explanation_style": p_obj.explanation_style,
                        "learning_goals": p_obj.learning_goals or [],
                    }
            except Exception as e:
                logger.warning(f"Failed to load profile in MemoryService: {e}")

        # 2. Retrieve Relevant Semantic Memories
        if self.memory_repo:
            try:
                m_objs = await self.memory_repo.search_memories(user_id=user_id, query=query, limit=5)
                semantic_memories = [
                    {
                        "id": m.id,
                        "topic": m.topic,
                        "content": m.content,
                        "memory_type": m.memory_type,
                        "importance": m.importance,
                    }
                    for m in m_objs
                ]
            except Exception as e:
                logger.warning(f"Failed to search semantic memories: {e}")

        # 3. Retrieve Relevant Past Episodes
        if self.episodic_repo:
            try:
                e_objs = await self.episodic_repo.search_episodes(user_id=user_id, query=query, limit=3)
                episodic_memories = [
                    {
                        "id": e.id,
                        "event_type": e.event_type,
                        "topic": e.topic,
                        "summary": e.summary,
                        "importance": e.importance,
                    }
                    for e in e_objs
                ]
            except Exception as e:
                logger.warning(f"Failed to search episodic memories: {e}")

        return {
            "profile": profile_data,
            "semantic_memories": semantic_memories,
            "episodic_memories": episodic_memories,
        }

    async def extract_and_update_memories(
        self,
        user_id: str,
        query: str,
        final_answer: str,
        study_artifacts: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Post-execution memory update:
        Identifies new durable preferences and meaningful learning episodes (e.g. quiz outcomes),
        validates them against anti-hallucination rules, and persists through controlled repositories.
        """
        extracted_semantic: List[Dict[str, Any]] = []
        extracted_episodic: List[Dict[str, Any]] = []

        if not user_id or not query:
            return {"semantic": [], "episodic": []}

        # 1. Automatic episodic record for study artifacts (MCQ / Flashcard sessions)
        if study_artifacts and self.episodic_repo:
            try:
                count = len(study_artifacts)
                summary = f"Generated and completed practice study session with {count} items on '{query[:60]}'."
                ep = await self.episodic_repo.add_episode(
                    user_id=user_id,
                    event_type="revision_milestone",
                    topic=query[:100],
                    summary=summary,
                    details_json={"artifact_count": count},
                    session_id=session_id,
                    importance=2,
                )
                extracted_episodic.append({"id": ep.id, "summary": ep.summary, "type": ep.event_type})
            except Exception as e:
                logger.warning(f"Failed to auto-record study episode: {e}")

        # 2. Rule-based fast check for explicit memory triggers
        lower_q = query.lower()
        has_explicit_trigger = any(
            t in lower_q
            for t in [
                "remember this",
                "save this topic",
                "add to my goals",
                "my goal is",
                "i want to learn",
                "prefer explanations",
                "my preferred style",
            ]
        )

        if not has_explicit_trigger:
            return {"semantic": extracted_semantic, "episodic": extracted_episodic}

        # 3. LLM memory extraction with validation prompt
        messages = [
            LLMMessage(role="system", content=MEMORY_EXTRACTION_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"Student Message: {query}\n\n"
                    f"Assistant Response Summary: {final_answer[:400]}"
                ),
            ),
        ]

        try:
            res = await self.router.execute_with_fallback(
                task_type=TaskType.CLASSIFICATION,
                messages=messages,
                complexity=ComplexityLevel.LOW,
                temperature=0.1,
                max_tokens=300,
            )

            raw = res.content.strip()
            clean_json = re.sub(r"^```json\s*", "", raw)
            clean_json = re.sub(r"\s*```$", "", clean_json)
            parsed = json.loads(clean_json)

            # Persist semantic memories
            if self.memory_repo:
                for sm in parsed.get("semantic_memories", []):
                    content = sm.get("content", "").strip()
                    if content and len(content) > 5:
                        mem = await self.memory_repo.add_memory(
                            user_id=user_id,
                            memory_type=sm.get("memory_type", "preference"),
                            content=content,
                            topic=sm.get("topic") or "General",
                            importance=sm.get("importance", 1),
                        )
                        extracted_semantic.append({
                            "id": mem.id,
                            "topic": mem.topic,
                            "content": mem.content,
                            "type": mem.memory_type,
                        })

            # Persist episodic memories
            if self.episodic_repo:
                for em in parsed.get("episodic_memories", []):
                    summary = em.get("summary", "").strip()
                    if summary:
                        ep = await self.episodic_repo.add_episode(
                            user_id=user_id,
                            event_type=em.get("event_type", "revision_milestone"),
                            topic=em.get("topic") or query[:50],
                            summary=summary,
                            session_id=session_id,
                            importance=em.get("importance", 1),
                        )
                        extracted_episodic.append({
                            "id": ep.id,
                            "topic": ep.topic,
                            "summary": ep.summary,
                            "type": ep.event_type,
                        })

        except Exception as e:
            logger.warning(f"LLM memory extraction failed: {e}")

        return {"semantic": extracted_semantic, "episodic": extracted_episodic}

    async def delete_memory(self, memory_id: str, user_id: str, memory_type: str = "semantic") -> bool:
        """Securely delete a memory record enforcing tenant isolation."""
        if memory_type == "episodic" and self.episodic_repo:
            return await self.episodic_repo.delete_episode(memory_id, user_id)
        elif self.memory_repo:
            return await self.memory_repo.delete_memory(memory_id, user_id)
        return False
