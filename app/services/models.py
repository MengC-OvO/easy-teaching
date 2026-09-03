"""SQLAlchemy ORM models for EasyTeaching-owned PostgreSQL tables."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.schemas.long_memory import MemoryRetrievalMode


class Base(DeclarativeBase):
    pass


class CentreRecord(Base):
    __tablename__ = "centres"

    centre_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    suburb: Mapped[str] = mapped_column(String, nullable=False)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timezone: Mapped[str] = mapped_column(
        String, nullable=False, default="Australia/Sydney"
    )
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TeacherRecord(Base):
    __tablename__ = "teachers"

    teacher_id: Mapped[str] = mapped_column(String, primary_key=True)
    auth_user_id: Mapped[Optional[str]] = mapped_column(
        String, unique=True, nullable=True, index=True
    )
    centre_id: Mapped[str] = mapped_column(
        ForeignKey("centres.centre_id"), nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ClassRecord(Base):
    __tablename__ = "classes"

    class_id: Mapped[str] = mapped_column(String, primary_key=True)
    centre_id: Mapped[str] = mapped_column(
        ForeignKey("centres.centre_id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    age_group: Mapped[str] = mapped_column(String, nullable=False)
    current_focus: Mapped[List[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TeacherClassMembershipRecord(Base):
    __tablename__ = "teacher_class_memberships"
    __table_args__ = (
        UniqueConstraint(
            "teacher_id", "class_id", name="uq_teacher_class_membership"
        ),
    )

    membership_id: Mapped[str] = mapped_column(String, primary_key=True)
    teacher_id: Mapped[str] = mapped_column(
        ForeignKey("teachers.teacher_id"), nullable=False, index=True
    )
    class_id: Mapped[str] = mapped_column(
        ForeignKey("classes.class_id"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String, nullable=False, default="educator")
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )


class ChildRecord(Base):
    __tablename__ = "children"

    child_id: Mapped[str] = mapped_column(String, primary_key=True)
    class_id: Mapped[str] = mapped_column(
        ForeignKey("classes.class_id"), nullable=False, index=True
    )
    display_code: Mapped[str] = mapped_column(String, nullable=False, index=True)
    preferred_name_encrypted: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    date_of_birth_encrypted: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


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


class ObservationRecord(Base):
    __tablename__ = "observations"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    centre_id: Mapped[str] = mapped_column(
        ForeignKey("centres.centre_id"), nullable=False, index=True
    )
    class_id: Mapped[str] = mapped_column(
        ForeignKey("classes.class_id"), nullable=False, index=True
    )
    author_teacher_id: Mapped[str] = mapped_column(
        ForeignKey("teachers.teacher_id"), nullable=False, index=True
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    setting: Mapped[str] = mapped_column(String, nullable=False)
    objective_text: Mapped[str] = mapped_column(Text, nullable=False)
    educator_actions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="draft", index=True
    )
    source_request_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(
        String, nullable=False, unique=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ConversationTaskOutboxRecord(Base):
    """Durable hand-off from the API transaction to the Celery broker.

    The broker message contains only ``request_id``.  The redacted execution
    payload remains in PostgreSQL so a broker outage cannot lose accepted work.
    """

    __tablename__ = "conversation_task_outbox"

    request_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_runs.request_id", ondelete="CASCADE"),
        primary_key=True,
    )
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    publish_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    execution_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    lease_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ObservationChildRecord(Base):
    __tablename__ = "observation_children"
    __table_args__ = (
        UniqueConstraint(
            "observation_id", "child_id", name="uq_observation_child"
        ),
    )

    link_id: Mapped[str] = mapped_column(String, primary_key=True)
    observation_id: Mapped[str] = mapped_column(
        ForeignKey("observations.observation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    child_id: Mapped[str] = mapped_column(
        ForeignKey("children.child_id"), nullable=False, index=True
    )


class EducationalRecord(Base):
    __tablename__ = "educational_records"

    record_id: Mapped[str] = mapped_column(String, primary_key=True)
    centre_id: Mapped[str] = mapped_column(
        ForeignKey("centres.centre_id"), nullable=False, index=True
    )
    class_id: Mapped[str] = mapped_column(
        ForeignKey("classes.class_id"), nullable=False, index=True
    )
    author_teacher_id: Mapped[str] = mapped_column(
        ForeignKey("teachers.teacher_id"), nullable=False, index=True
    )
    record_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    analysis: Mapped[str] = mapped_column(Text, nullable=False)
    curriculum_links: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    next_steps: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="draft", index=True
    )
    source_request_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(
        String, nullable=False, unique=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    approved_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class EducationalRecordObservation(Base):
    __tablename__ = "educational_record_observations"
    __table_args__ = (
        UniqueConstraint(
            "record_id", "observation_id", name="uq_educational_record_observation"
        ),
    )

    link_id: Mapped[str] = mapped_column(String, primary_key=True)
    record_id: Mapped[str] = mapped_column(
        ForeignKey("educational_records.record_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    observation_id: Mapped[str] = mapped_column(
        ForeignKey("observations.observation_id"), nullable=False, index=True
    )


class RecordExport(Base):
    __tablename__ = "record_exports"

    export_id: Mapped[str] = mapped_column(String, primary_key=True)
    teacher_id: Mapped[str] = mapped_column(
        ForeignKey("teachers.teacher_id"), nullable=False, index=True
    )
    record_ids: Mapped[List[str]] = mapped_column(JSON, nullable=False)
    format: Mapped[str] = mapped_column(String, nullable=False)
    template_name: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="ready", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class ToolActionRequest(Base):
    __tablename__ = "tool_action_requests"

    action_id: Mapped[str] = mapped_column(String, primary_key=True)
    request_id: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    teacher_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    class_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    tool_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    arguments: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    arguments_hash: Mapped[str] = mapped_column(String, nullable=False)
    preview: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending", index=True
    )
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )


class KnowledgeSourceRecord(Base):
    __tablename__ = "knowledge_sources"

    source_id: Mapped[str] = mapped_column(String, primary_key=True)
    source_key: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String, nullable=False)
    effective_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    checksum: Mapped[str] = mapped_column(String, nullable=False)
    knowledge_scope: Mapped[str] = mapped_column(String, nullable=False, index=True)
    index_version: Mapped[str] = mapped_column(String, nullable=False)
    index_status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending", index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class AuditEventRecord(Base):
    __tablename__ = "audit_events"

    audit_id: Mapped[str] = mapped_column(String, primary_key=True)
    teacher_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    class_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    tool_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    result: Mapped[str] = mapped_column(String, nullable=False)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True
    )


class LongTermMemoryRecord(Base):
    """A reviewed, durable memory scoped to one teacher or class."""

    __tablename__ = "long_term_memories"

    memory_id: Mapped[str] = mapped_column(String, primary_key=True)
    centre_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
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
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    review_status: Mapped[str] = mapped_column(
        String, nullable=False, default="auto", index=True
    )
    source_request_id: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, index=True
    )
    last_confirmed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
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
