import uuid
from typing import Optional, Dict, Any, TYPE_CHECKING
from sqlalchemy import String, ForeignKey, Integer, Text, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.db.base import Base, TimestampMixin
from app.core.config import settings

if TYPE_CHECKING:
    from app.db.models.user import User


class EpisodicMemory(Base, TimestampMixin):
    """
    Episodic learning memory storing discrete past learning events:
    - Quiz completion and score outcomes (e.g. scored 3/5 on endodontic microbiology)
    - Specific concepts struggled with or mastered
    - Study session milestones and topic revisions
    """
    __tablename__ = "episodic_memories"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    session_id: Mapped[Optional[str]] = mapped_column(String(36), index=True, nullable=True)
    event_type: Mapped[str] = mapped_column(
        String(50), index=True, nullable=False
    )  # "quiz_completed", "struggled_concept", "mastered_concept", "revision_milestone", "session_summary"
    topic: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict, nullable=True)
    importance: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    embedding: Mapped[Optional[Any]] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION), nullable=True
    )

    user: Mapped["User"] = relationship("User", back_populates="episodic_memories")

    __table_args__ = (
        Index("ix_episodic_memories_user_type", "user_id", "event_type"),
        Index("ix_episodic_memories_user_topic", "user_id", "topic"),
    )
