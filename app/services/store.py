from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    JSON,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.schemas.long_memory import (
    LongTermMemoryAction,
    LongTermMemoryCandidate,
    LongTermMemoryOperation,
    MemoryRetrievalMode,
)


SYNTHETIC_CLASS_PROFILES = [
    {
        "class_id": "kangaroo-room",
        "name": "Kangaroo Room",
        "age_group": "3-5",
        "child_count": 18,
        "interests": ["outdoor play", "storytelling", "sensory exploration"],
        "safety_notes": ["synthetic data only", "check allergies before food play"],
    }
]

ACTIVE_CONVERSATION_RUN_STATUSES = (
    "accepted",
    "running",
    "waiting_for_approval",
)


class ConversationSessionBusyError(RuntimeError):
    """Raised when another active run wins the session-level race."""

    def __init__(self, active_request_id: str) -> None:
        super().__init__("Conversation session already has an active run")
        self.active_request_id = active_request_id


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
    idempotency_key: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
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


class EduFlowStore:
    def __init__(self, database_url: Optional[str] = None) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self.engine = create_engine(self.database_url, future=True)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            future=True,
        )

    def initialize(self, *, create_schema: bool = True) -> None:
        """Prepare the store; production PostgreSQL schema ownership is Alembic."""
        if create_schema:
            Base.metadata.create_all(self.engine)
        self._ensure_active_conversation_run_index()
        with self.session_factory() as session:
            self._seed_class_profiles(session)
            session.commit()

    def get_class_profile(self, class_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            profile = session.get(ClassProfile, class_id)
            if profile is None:
                return None
            return self._class_profile_to_dict(profile)

    def create_conversation_session(
        self,
        *,
        session_id: str,
        thread_id: str,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> Dict[str, Any]:
        record = ConversationSessionRecord(
            session_id=session_id,
            thread_id=thread_id,
            teacher_id=teacher_id,
            class_id=class_id,
            status="active",
        )
        with self.session_factory() as session:
            session.add(record)
            session.commit()
        return self._conversation_session_to_dict(record)

    def get_conversation_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            record = session.get(ConversationSessionRecord, session_id)
            if record is None:
                return None
            return self._conversation_session_to_dict(record)

    def create_conversation_run(
        self,
        *,
        request_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        with self.session_factory() as session:
            existing = session.get(ConversationRunRecord, request_id)
            if existing is not None:
                return self._conversation_run_to_dict(existing, created=False)

            record = ConversationRunRecord(
                request_id=request_id,
                session_id=session_id,
                status="accepted",
            )
            session.add(record)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                existing = session.get(ConversationRunRecord, request_id)
                if existing is not None:
                    return self._conversation_run_to_dict(existing, created=False)
                active = self._active_conversation_run(session, session_id)
                if active is not None:
                    raise ConversationSessionBusyError(active.request_id)
                raise
        return self._conversation_run_to_dict(record, created=True)

    def get_conversation_run(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            record = session.get(ConversationRunRecord, request_id)
            if record is None:
                return None
            return self._conversation_run_to_dict(record, created=False)

    def get_active_conversation_run(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            record = self._active_conversation_run(session, session_id)
            if record is None:
                return None
            return self._conversation_run_to_dict(record, created=False)

    def list_conversation_runs(
        self,
        *,
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            statement = select(ConversationRunRecord).order_by(
                ConversationRunRecord.created_at
            )
            if statuses:
                statement = statement.where(
                    ConversationRunRecord.status.in_(statuses)
                )
            records = session.execute(statement).scalars().all()
        return [
            self._conversation_run_to_dict(record, created=False)
            for record in records
        ]

    def update_conversation_run_status(
        self,
        request_id: str,
        status: str,
    ) -> Dict[str, Any]:
        with self.session_factory() as session:
            record = session.get(ConversationRunRecord, request_id)
            if record is None:
                raise ValueError("Conversation run does not exist")
            record.status = status
            record.updated_at = datetime.utcnow()
            session.commit()
        return self._conversation_run_to_dict(record, created=False)

    def save_conversation_run_result(
        self,
        *,
        request_id: str,
        session_id: str,
        draft: Dict[str, Any],
        approval: Dict[str, Any],
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        with self.session_factory() as session:
            record = session.get(ConversationRunResultRecord, request_id)
            if record is None:
                record = ConversationRunResultRecord(
                    request_id=request_id,
                    session_id=session_id,
                    draft=draft,
                    approval=approval,
                    citations=citations,
                )
                session.add(record)
            else:
                if record.session_id != session_id:
                    raise ValueError("Conversation run result belongs to another session")
                record.draft = draft
                record.approval = approval
                record.citations = citations
            session.commit()
        return self._conversation_run_result_to_dict(record)

    def get_conversation_run_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            record = session.get(ConversationRunResultRecord, request_id)
            if record is None:
                return None
            return self._conversation_run_result_to_dict(record)

    def append_conversation_event(
        self,
        *,
        request_id: str,
        session_id: str,
        event: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self.session_factory() as session:
            last_sequence = session.execute(
                select(func.max(ConversationEventRecord.sequence)).where(
                    ConversationEventRecord.request_id == request_id
                )
            ).scalar_one()
            sequence = 0 if last_sequence is None else last_sequence + 1
            record = ConversationEventRecord(
                event_id=str(uuid4()),
                request_id=request_id,
                session_id=session_id,
                event=event,
                sequence=sequence,
                data=data,
            )
            session.add(record)
            session.commit()
        return self._conversation_event_to_dict(record)

    def list_conversation_events(
        self,
        *,
        request_id: str,
        after_sequence: int = -1,
    ) -> List[Dict[str, Any]]:
        with self.session_factory() as session:
            records = (
                session.execute(
                    select(ConversationEventRecord)
                    .where(
                        ConversationEventRecord.request_id == request_id,
                        ConversationEventRecord.sequence > after_sequence,
                    )
                    .order_by(ConversationEventRecord.sequence)
                )
                .scalars()
                .all()
            )
        return [self._conversation_event_to_dict(record) for record in records]

    def save_long_term_memory(
        self,
        candidate: LongTermMemoryCandidate,
    ) -> Dict[str, str]:
        memory = LongTermMemoryRecord(
            memory_id=str(uuid4()),
            scope=candidate.scope.value,
            scope_id=candidate.scope_id,
            memory_type=candidate.memory_type.value,
            content=candidate.content,
            reason=candidate.reason,
            retrieval_mode=candidate.retrieval_mode.value,
            importance=candidate.importance,
            is_active=True,
        )
        with self.session_factory() as session:
            session.add(memory)
            session.commit()
        return self._long_term_memory_to_dict(memory)

    def apply_long_term_memory_operation(
        self,
        operation: LongTermMemoryOperation,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> Dict[str, str]:
        """Apply one validated LLM decision without allowing cross-owner writes."""
        if operation.action is LongTermMemoryAction.NOOP:
            return {"action": operation.action.value}
        if operation.action is LongTermMemoryAction.INSERT:
            assert operation.candidate is not None
            self._validate_memory_owner(operation.candidate, teacher_id, class_id)
            memory = self.save_long_term_memory(operation.candidate)
            return {"action": operation.action.value, **memory}

        assert operation.memory_id is not None
        with self.session_factory() as session:
            memory = session.get(LongTermMemoryRecord, operation.memory_id)
            if memory is None:
                raise ValueError("Long-term memory does not exist")
            self._validate_record_owner(memory, teacher_id, class_id)

            if operation.action is LongTermMemoryAction.DELETE:
                memory.is_active = False
                memory.updated_at = datetime.utcnow()
                session.commit()
                return {
                    "action": operation.action.value,
                    "memory_id": operation.memory_id,
                }

            assert operation.action is LongTermMemoryAction.UPDATE
            assert operation.candidate is not None
            candidate = operation.candidate
            if (
                memory.scope != candidate.scope.value
                or memory.scope_id != candidate.scope_id
            ):
                raise ValueError("Long-term memory update cannot change its owner")
            memory.memory_type = candidate.memory_type.value
            memory.content = candidate.content
            memory.reason = candidate.reason
            memory.retrieval_mode = candidate.retrieval_mode.value
            memory.importance = candidate.importance
            memory.is_active = True
            memory.updated_at = datetime.utcnow()
            session.commit()
            return {"action": operation.action.value, **self._long_term_memory_to_dict(memory)}

    def list_long_term_memories(
        self,
        *,
        scope: str,
        scope_id: str,
    ) -> List[Dict[str, str]]:
        with self.session_factory() as session:
            memories = session.execute(
                select(LongTermMemoryRecord)
                .where(
                    LongTermMemoryRecord.scope == scope,
                    LongTermMemoryRecord.scope_id == scope_id,
                    LongTermMemoryRecord.is_active.is_(True),
                )
                .order_by(LongTermMemoryRecord.memory_id)
            ).scalars().all()
        return [self._long_term_memory_to_dict(memory) for memory in memories]

    def list_profile_memories(
        self,
        *,
        teacher_id: Optional[str],
        limit: int = 4,
    ) -> List[Dict[str, str]]:
        """Load stable teacher preferences safe to include in every task prompt."""
        if not teacher_id:
            return []
        with self.session_factory() as session:
            memories = session.execute(
                select(LongTermMemoryRecord)
                .where(
                    LongTermMemoryRecord.scope == "teacher",
                    LongTermMemoryRecord.scope_id == teacher_id,
                    LongTermMemoryRecord.memory_type == "teacher_preference",
                    LongTermMemoryRecord.retrieval_mode
                    == MemoryRetrievalMode.PROFILE.value,
                    LongTermMemoryRecord.is_active.is_(True),
                )
                .order_by(
                    LongTermMemoryRecord.importance.desc(),
                    LongTermMemoryRecord.updated_at.desc(),
                )
                .limit(limit)
            ).scalars().all()
        return [self._long_term_memory_to_dict(memory) for memory in memories]

    def list_memories_for_owners(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        limit: int = 12,
    ) -> List[Dict[str, str]]:
        """Return only memory records that belong to the active request owners."""
        memories: List[Dict[str, str]] = []
        if teacher_id:
            memories.extend(
                self.list_long_term_memories(scope="teacher", scope_id=teacher_id)
            )
        if class_id:
            memories.extend(
                self.list_long_term_memories(scope="class", scope_id=class_id)
            )
        return memories[-limit:]

    def search_recall_memories(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, str]]:
        """Search on-demand memories only, scoped to the active teacher/class."""
        candidates = self.list_memories_for_owners(
            teacher_id=teacher_id,
            class_id=class_id,
            limit=100,
        )
        query_tokens = [token.lower() for token in query.split() if token]
        recalled = [
            memory
            for memory in candidates
            if memory["retrieval_mode"] == MemoryRetrievalMode.RECALL_ONLY.value
            and (
                not query_tokens
                or any(token in memory["content"].lower() for token in query_tokens)
            )
        ]
        return recalled[:limit]

    def count_class_profiles(self) -> int:
        with self.session_factory() as session:
            return len(session.execute(select(ClassProfile)).scalars().all())

    def _seed_class_profiles(self, session: Session) -> None:
        for profile in SYNTHETIC_CLASS_PROFILES:
            if session.get(ClassProfile, profile["class_id"]) is not None:
                continue
            session.add(ClassProfile(**profile))

    def _active_conversation_run(
        self,
        session: Session,
        session_id: str,
    ) -> Optional[ConversationRunRecord]:
        return (
            session.execute(
                select(ConversationRunRecord)
                .where(
                    ConversationRunRecord.session_id == session_id,
                    ConversationRunRecord.status.in_(
                        ACTIVE_CONVERSATION_RUN_STATUSES
                    ),
                )
                .order_by(ConversationRunRecord.created_at.desc())
            )
            .scalars()
            .first()
        )

    def _ensure_active_conversation_run_index(self) -> None:
        """Install the partial unique index for existing local databases too."""
        index = next(
            item
            for item in ConversationRunRecord.__table__.indexes
            if item.name == "uq_conversation_runs_one_active_per_session"
        )
        index.create(self.engine, checkfirst=True)

    def _class_profile_to_dict(self, profile: ClassProfile) -> Dict[str, Any]:
        return {
            "class_id": profile.class_id,
            "name": profile.name,
            "age_group": profile.age_group,
            "child_count": profile.child_count,
            "interests": profile.interests,
            "safety_notes": profile.safety_notes,
        }

    def _conversation_session_to_dict(
        self,
        record: ConversationSessionRecord,
    ) -> Dict[str, Any]:
        return {
            "session_id": record.session_id,
            "thread_id": record.thread_id,
            "teacher_id": record.teacher_id,
            "class_id": record.class_id,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
        }

    def _conversation_run_to_dict(
        self,
        record: ConversationRunRecord,
        *,
        created: bool,
    ) -> Dict[str, Any]:
        return {
            "request_id": record.request_id,
            "session_id": record.session_id,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "created": created,
        }

    def _conversation_run_result_to_dict(
        self,
        record: ConversationRunResultRecord,
    ) -> Dict[str, Any]:
        return {
            "request_id": record.request_id,
            "session_id": record.session_id,
            "draft": record.draft,
            "approval": record.approval,
            "citations": record.citations,
            "created_at": record.created_at.isoformat(),
        }

    def _conversation_event_to_dict(
        self,
        record: ConversationEventRecord,
    ) -> Dict[str, Any]:
        return {
            "event_id": record.event_id,
            "request_id": record.request_id,
            "session_id": record.session_id,
            "event": record.event,
            "sequence": record.sequence,
            "data": record.data,
            "created_at": record.created_at.isoformat(),
        }

    def _long_term_memory_to_dict(
        self,
        memory: LongTermMemoryRecord,
    ) -> Dict[str, str]:
        return {
            "memory_id": memory.memory_id,
            "scope": memory.scope,
            "scope_id": memory.scope_id,
            "memory_type": memory.memory_type,
            "content": memory.content,
            "reason": memory.reason,
            "retrieval_mode": memory.retrieval_mode,
            "importance": str(memory.importance),
            "is_active": str(memory.is_active).lower(),
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
        }

    def _validate_memory_owner(
        self,
        candidate: LongTermMemoryCandidate,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> None:
        if candidate.scope.value == "teacher" and candidate.scope_id != teacher_id:
            raise ValueError("Long-term memory operation is outside the active teacher")
        if candidate.scope.value == "class" and candidate.scope_id != class_id:
            raise ValueError("Long-term memory operation is outside the active class")

    def _validate_record_owner(
        self,
        memory: LongTermMemoryRecord,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> None:
        if memory.scope == "teacher" and memory.scope_id != teacher_id:
            raise ValueError("Long-term memory operation is outside the active teacher")
        if memory.scope == "class" and memory.scope_id != class_id:
            raise ValueError("Long-term memory operation is outside the active class")
