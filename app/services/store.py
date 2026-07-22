from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, JSON, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import settings
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


class DraftRecord(Base):
    __tablename__ = "drafts"

    draft_id: Mapped[str] = mapped_column(String, primary_key=True)
    idempotency_key: Mapped[Optional[str]] = mapped_column(String, unique=True, nullable=True)
    draft_type: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="draft")


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
        self.database_url = database_url or self._sqlite_url(settings.database_path)
        self.engine = create_engine(self.database_url, future=True)
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
            future=True,
        )

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        self._migrate_long_term_memory_schema()
        with self.session_factory() as session:
            self._seed_class_profiles(session)
            session.commit()

    def get_class_profile(self, class_id: str) -> Optional[Dict[str, Any]]:
        with self.session_factory() as session:
            profile = session.get(ClassProfile, class_id)
            if profile is None:
                return None
            return self._class_profile_to_dict(profile)

    def save_draft(
        self,
        *,
        draft_id: str,
        draft_type: str,
        title: str,
        content: str,
        status: str = "draft",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, str]:
        with self.session_factory() as session:
            if idempotency_key:
                existing = (
                    session.execute(
                        select(DraftRecord).where(
                            DraftRecord.idempotency_key == idempotency_key
                        )
                    )
                    .scalars()
                    .first()
                )
                if existing is not None:
                    return self._draft_to_dict(existing)

            draft = DraftRecord(
                draft_id=draft_id,
                idempotency_key=idempotency_key,
                draft_type=draft_type,
                title=title,
                content=content,
                status=status,
            )
            session.add(draft)
            session.commit()

        return self._draft_to_dict(draft)

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

    def _sqlite_url(self, database_path: str) -> str:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path}"

    def _class_profile_to_dict(self, profile: ClassProfile) -> Dict[str, Any]:
        return {
            "class_id": profile.class_id,
            "name": profile.name,
            "age_group": profile.age_group,
            "child_count": profile.child_count,
            "interests": profile.interests,
            "safety_notes": profile.safety_notes,
        }

    def _draft_to_dict(self, draft: DraftRecord) -> Dict[str, str]:
        return {
            "draft_id": draft.draft_id,
            "draft_type": draft.draft_type,
            "title": draft.title,
            "status": draft.status,
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

    def _migrate_long_term_memory_schema(self) -> None:
        """Add Day 6 profile/recall fields to an existing local SQLite database."""
        with self.engine.connect() as connection:
            columns = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(long_term_memories)"
                ).fetchall()
            }
        additions = {
            "retrieval_mode": "VARCHAR NOT NULL DEFAULT 'recall_only'",
            "importance": "INTEGER NOT NULL DEFAULT 2",
            "is_active": "BOOLEAN NOT NULL DEFAULT 1",
            "created_at": "DATETIME",
            "updated_at": "DATETIME",
        }
        with self.engine.begin() as connection:
            for column, definition in additions.items():
                if column not in columns:
                    connection.exec_driver_sql(
                        f"ALTER TABLE long_term_memories ADD COLUMN {column} {definition}"
                    )
            connection.exec_driver_sql(
                "UPDATE long_term_memories "
                "SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
            )
            connection.exec_driver_sql(
                "UPDATE long_term_memories "
                "SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL"
            )

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
