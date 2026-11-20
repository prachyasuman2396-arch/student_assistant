from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, delete
from app.db.models.episodic_memory import EpisodicMemory
from app.core.logging import logger


class EpisodicRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_episode(
        self,
        user_id: str,
        event_type: str,
        topic: str,
        summary: str,
        details_json: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        importance: int = 1,
        embedding: Optional[List[float]] = None,
    ) -> EpisodicMemory:
        """
        Record a meaningful past learning episode (quiz outcome, concept struggle, revision milestone).
        """
        episode = EpisodicMemory(
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            topic=topic.strip(),
            summary=summary.strip(),
            details_json=details_json or {},
            importance=importance,
            embedding=embedding,
        )
        self.session.add(episode)
        await self.session.commit()
        await self.session.refresh(episode)
        return episode

    async def get_recent_episodes(
        self,
        user_id: str,
        event_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[EpisodicMemory]:
        """Fetch user's recent learning episodes ordered by recency and importance."""
        stmt = select(EpisodicMemory).where(EpisodicMemory.user_id == user_id)
        if event_type:
            stmt = stmt.where(EpisodicMemory.event_type == event_type)
        stmt = stmt.order_by(EpisodicMemory.importance.desc(), EpisodicMemory.created_at.desc()).limit(limit)
        res = await self.session.execute(stmt)
        return list(res.scalars().all())

    async def search_episodes(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[EpisodicMemory]:
        """Keyword relevance search across topic and summary for user's past episodes."""
        query_terms = [t.lower() for t in query.split() if len(t) > 2]
        all_episodes = await self.get_recent_episodes(user_id=user_id, limit=30)
        if not query_terms:
            return all_episodes[:limit]

        scored: List[tuple[int, EpisodicMemory]] = []
        for ep in all_episodes:
            text = f"{ep.topic} {ep.summary} {ep.event_type}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, ep))

        scored.sort(key=lambda x: (x[0], x[1].importance), reverse=True)
        return [item[1] for item in scored[:limit]]

    async def delete_episode(self, episode_id: str, user_id: str) -> bool:
        """Delete specific episode ensuring strict user isolation."""
        stmt = delete(EpisodicMemory).where(
            and_(
                EpisodicMemory.id == episode_id,
                EpisodicMemory.user_id == user_id,
            )
        )
        res = await self.session.execute(stmt)
        await self.session.commit()
        return res.rowcount > 0
