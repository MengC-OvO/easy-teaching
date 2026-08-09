"""SQLAlchemy ORM models for EduFlow-owned PostgreSQL tables."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    JSON,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.schemas.long_memory import MemoryRetrievalMode


class Base(DeclarativeBase):
    pass


class ClassProfile(Base):
    __tablename__ = "class_profiles"

    class_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    age_group: Mapped[str] = mapped_column(String, nullable=False)
    child_count: Mapped[int] = mapped_column(Integer, nullable=False)
    interests: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    safety_notes: Mapped[List[str]] = mapped_column(JSON, nullable=False)


class ConversationSessionRecord(Base):
    """Durable API session metadata; graph state remains in the checkpointer."""

    __tablename__ = "conversation_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    thread_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    teacher_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    class_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )


class ConversationRunRecord(Base):
    """Operational status for one message; content stays in graph checkpoints."""

    __tablename__ = "conversation_runs"
    __table_args__ = (
        Index(
            "uq_conversation_runs_one_active_per_session",
            "session_id",
            unique=True,
            postgresql_where=text(
                "status IN ('accepted', 'running', 'waiting_for_approval')"
            ),
        ),
    )

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


class ConversationRunResultRecord(Base):
    """Public draft snapshot produced by one conversation run."""

    __tablename__ = "conversation_run_results"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    draft: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    approval: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    citations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )


class ConversationApprovalDecisionRecord(Base):
    """Idempotency record for one human approval decision."""

    __tablename__ = "conversation_approval_decisions"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    decision: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )


class ConversationEventRecord(Base):
    """Ordered, replayable server-sent event for one conversation run."""

    __tablename__ = "conversation_events"
    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "sequence",
            name="uq_conversation_event_request_sequence",
        ),
    )

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    request_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )


class DraftRecord(Base):
    __tablename__ = "drafts"

    draft_id: Mapped[str] = mapped_column(String, primary_key=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String,
        unique=True,
        nullable=True,
    )
    draft_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft")


class LearningRecord(Base):
    """Teacher-approved record; the original observation is never stored here."""

    __tablename__ = "learning_records"

    record_id: Mapped[str] = mapped_column(String, primary_key=True)
    request_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    teacher_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    class_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="approved")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )


class LongTermMemoryRecord(Base):
    """A reviewed, durable memory scoped to one teacher or class."""

    __tablename__ = "long_term_memories"

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    scope: Mapped[str] = mapped_column(String, nullable=False, index=True)
    scope_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    memory_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    content: Mapped[str] = mapped_column(String, nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default=MemoryRetrievalMode.RECALL_ONLY.value,
        index=True,
    )
    importance: Mapped[int] = mapped_column(Integer, nullable=False, default=2, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
