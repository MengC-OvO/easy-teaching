"""PostgreSQL-only asynchronous persistence used by the production API."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.schemas.long_memory import (
    LongTermMemoryAction,
    LongTermMemoryOperation,
    MemoryRetrievalMode,
)
from app.services.models import (
    ClassProfile,
    ConversationEventRecord,
    ConversationRunRecord,
    ConversationRunResultRecord,
    ConversationSessionRecord,
    LongTermMemoryRecord,
)


SYNTHETIC_CLASS_PROFILES = (
    {
        "class_id": "kangaroo-room",
        "name": "Kangaroo Room",
        "age_group": "3-5",
        "child_count": 18,
        "interests": ["outdoor play", "storytelling", "sensory exploration"],
        "safety_notes": [
            "synthetic data only",
            "check allergies before food play",
        ],
    },
)

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


class AsyncEduFlowStore:
    """Async SQLAlchemy store; production schema ownership remains with Alembic."""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
            raise ValueError("Production DATABASE_URL must use PostgreSQL")
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        self.database_url = database_url
        self.engine = create_async_engine(database_url, pool_pre_ping=True)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        async with self.session_factory() as session:
            for profile in SYNTHETIC_CLASS_PROFILES:
                if await session.get(ClassProfile, profile["class_id"]) is None:
                    session.add(ClassProfile(**profile))
            await session.commit()

    async def close(self) -> None:
        await self.engine.dispose()

    async def get_class_profile(self, class_id: str) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            profile = await session.get(ClassProfile, class_id)
            return None if profile is None else self._class_profile_to_dict(profile)

    async def create_conversation_session(
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
        async with self.session_factory() as session:
            session.add(record)
            await session.commit()
        return self._conversation_session_to_dict(record)

    async def get_conversation_session(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            record = await session.get(ConversationSessionRecord, session_id)
            return None if record is None else self._conversation_session_to_dict(record)

    async def create_conversation_run(
        self, *, request_id: str, session_id: str
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            existing = await session.get(ConversationRunRecord, request_id)
            if existing is not None:
                return self._conversation_run_to_dict(existing, created=False)
            record = ConversationRunRecord(
                request_id=request_id,
                session_id=session_id,
                status="accepted",
            )
            session.add(record)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await session.get(ConversationRunRecord, request_id)
                if existing is not None:
                    return self._conversation_run_to_dict(existing, created=False)
                active = await self._active_conversation_run(session, session_id)
                if active is not None:
                    raise ConversationSessionBusyError(active.request_id)
                raise
        return self._conversation_run_to_dict(record, created=True)

    async def get_conversation_run(
        self, request_id: str
    ) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            record = await session.get(ConversationRunRecord, request_id)
            return None if record is None else self._conversation_run_to_dict(
                record, created=False
            )

    async def get_active_conversation_run(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            record = await self._active_conversation_run(session, session_id)
            return None if record is None else self._conversation_run_to_dict(
                record, created=False
            )

    async def list_conversation_runs(
        self, *, statuses: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            statement = select(ConversationRunRecord).order_by(
                ConversationRunRecord.created_at
            )
            if statuses:
                statement = statement.where(ConversationRunRecord.status.in_(statuses))
            records = (await session.execute(statement)).scalars().all()
        return [self._conversation_run_to_dict(item, created=False) for item in records]

    async def update_conversation_run_status(
        self, request_id: str, status: str
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            record = await session.get(ConversationRunRecord, request_id)
            if record is None:
                raise ValueError("Conversation run does not exist")
            record.status = status
            record.updated_at = datetime.utcnow()
            await session.commit()
        return self._conversation_run_to_dict(record, created=False)

    async def save_conversation_run_result(
        self,
        *,
        request_id: str,
        session_id: str,
        draft: Dict[str, Any],
        approval: Dict[str, Any],
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            record = await session.get(ConversationRunResultRecord, request_id)
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
            await session.commit()
        return self._conversation_run_result_to_dict(record)

    async def get_conversation_run_result(
        self, request_id: str
    ) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            record = await session.get(ConversationRunResultRecord, request_id)
            return None if record is None else self._conversation_run_result_to_dict(record)

    async def append_conversation_event(
        self,
        *,
        request_id: str,
        session_id: str,
        event: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            last_sequence = (
                await session.execute(
                    select(func.max(ConversationEventRecord.sequence)).where(
                        ConversationEventRecord.request_id == request_id
                    )
                )
            ).scalar_one()
            record = ConversationEventRecord(
                event_id=str(uuid4()),
                request_id=request_id,
                session_id=session_id,
                event=event,
                sequence=0 if last_sequence is None else last_sequence + 1,
                data=data,
            )
            session.add(record)
            await session.commit()
        return self._conversation_event_to_dict(record)

    async def list_conversation_events(
        self, *, request_id: str, after_sequence: int = -1
    ) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            records = (
                (
                    await session.execute(
                        select(ConversationEventRecord)
                        .where(
                            ConversationEventRecord.request_id == request_id,
                            ConversationEventRecord.sequence > after_sequence,
                        )
                        .order_by(ConversationEventRecord.sequence)
                    )
                )
                .scalars()
                .all()
            )
        return [self._conversation_event_to_dict(item) for item in records]

    async def list_profile_memories(
        self, *, teacher_id: Optional[str], limit: int = 4
    ) -> List[Dict[str, str]]:
        if not teacher_id:
            return []
        async with self.session_factory() as session:
            records = (
                (
                    await session.execute(
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
                    )
                )
                .scalars()
                .all()
            )
        return [self._long_term_memory_to_dict(item) for item in records]

    async def list_memories_for_owners(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        limit: int = 12,
    ) -> List[Dict[str, str]]:
        owners = []
        if teacher_id:
            owners.append(("teacher", teacher_id))
        if class_id:
            owners.append(("class", class_id))
        if not owners:
            return []
        conditions = [
            (LongTermMemoryRecord.scope == scope)
            & (LongTermMemoryRecord.scope_id == scope_id)
            for scope, scope_id in owners
        ]
        from sqlalchemy import or_

        async with self.session_factory() as session:
            records = (
                (
                    await session.execute(
                        select(LongTermMemoryRecord)
                        .where(
                            or_(*conditions),
                            LongTermMemoryRecord.is_active.is_(True),
                        )
                        .order_by(LongTermMemoryRecord.updated_at.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [self._long_term_memory_to_dict(item) for item in reversed(records)]

    async def search_recall_memories(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, str]]:
        candidates = await self.list_memories_for_owners(
            teacher_id=teacher_id,
            class_id=class_id,
            limit=100,
        )
        tokens = [token.lower() for token in query.split() if token]
        return [
            item
            for item in candidates
            if item["retrieval_mode"] == MemoryRetrievalMode.RECALL_ONLY.value
            and (
                not tokens
                or any(token in item["content"].lower() for token in tokens)
            )
        ][:limit]

    async def apply_long_term_memory_operation(
        self,
        operation: LongTermMemoryOperation,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> Dict[str, str]:
        if operation.action is LongTermMemoryAction.NOOP:
            return {"action": operation.action.value}
        async with self.session_factory() as session:
            if operation.action is LongTermMemoryAction.INSERT:
                assert operation.candidate is not None
                candidate = operation.candidate
                self._validate_owner(candidate.scope.value, candidate.scope_id, teacher_id, class_id)
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
                session.add(memory)
            else:
                assert operation.memory_id is not None
                memory = await session.get(LongTermMemoryRecord, operation.memory_id)
                if memory is None:
                    raise ValueError("Long-term memory does not exist")
                self._validate_owner(memory.scope, memory.scope_id, teacher_id, class_id)
                if operation.action is LongTermMemoryAction.DELETE:
                    memory.is_active = False
                    memory.updated_at = datetime.utcnow()
                else:
                    assert operation.candidate is not None
                    candidate = operation.candidate
                    if memory.scope != candidate.scope.value or memory.scope_id != candidate.scope_id:
                        raise ValueError("Long-term memory update cannot change its owner")
                    memory.memory_type = candidate.memory_type.value
                    memory.content = candidate.content
                    memory.reason = candidate.reason
                    memory.retrieval_mode = candidate.retrieval_mode.value
                    memory.importance = candidate.importance
                    memory.is_active = True
                    memory.updated_at = datetime.utcnow()
            await session.commit()
        return {"action": operation.action.value, **self._long_term_memory_to_dict(memory)}

    async def _active_conversation_run(
        self, session: AsyncSession, session_id: str
    ) -> Optional[ConversationRunRecord]:
        return (
            (
                await session.execute(
                    select(ConversationRunRecord)
                    .where(
                        ConversationRunRecord.session_id == session_id,
                        ConversationRunRecord.status.in_(ACTIVE_CONVERSATION_RUN_STATUSES),
                    )
                    .order_by(ConversationRunRecord.created_at.desc())
                )
            )
            .scalars()
            .first()
        )

    @staticmethod
    def _validate_owner(scope, scope_id, teacher_id, class_id) -> None:
        if scope == "teacher" and scope_id != teacher_id:
            raise ValueError("Long-term memory operation is outside the active teacher")
        if scope == "class" and scope_id != class_id:
            raise ValueError("Long-term memory operation is outside the active class")

    @staticmethod
    def _class_profile_to_dict(profile: ClassProfile) -> Dict[str, Any]:
        return {
            "class_id": profile.class_id,
            "name": profile.name,
            "age_group": profile.age_group,
            "child_count": profile.child_count,
            "interests": profile.interests,
            "safety_notes": profile.safety_notes,
        }

    @staticmethod
    def _conversation_session_to_dict(record: ConversationSessionRecord) -> Dict[str, Any]:
        return {
            "session_id": record.session_id,
            "thread_id": record.thread_id,
            "teacher_id": record.teacher_id,
            "class_id": record.class_id,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
        }

    @staticmethod
    def _conversation_run_to_dict(record: ConversationRunRecord, *, created: bool) -> Dict[str, Any]:
        return {
            "request_id": record.request_id,
            "session_id": record.session_id,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "created": created,
        }

    @staticmethod
    def _conversation_run_result_to_dict(record: ConversationRunResultRecord) -> Dict[str, Any]:
        return {
            "request_id": record.request_id,
            "session_id": record.session_id,
            "draft": record.draft,
            "approval": record.approval,
            "citations": record.citations,
            "created_at": record.created_at.isoformat(),
        }

    @staticmethod
    def _conversation_event_to_dict(record: ConversationEventRecord) -> Dict[str, Any]:
        return {
            "event_id": record.event_id,
            "request_id": record.request_id,
            "session_id": record.session_id,
            "event": record.event,
            "sequence": record.sequence,
            "data": record.data,
            "created_at": record.created_at.isoformat(),
        }

    @staticmethod
    def _long_term_memory_to_dict(memory: LongTermMemoryRecord) -> Dict[str, str]:
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
