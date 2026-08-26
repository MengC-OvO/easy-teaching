"""PostgreSQL-only asynchronous persistence used by the production API."""

from datetime import datetime, timedelta
import hashlib
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy import func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.errors import ConversationSessionBusyError
from app.schemas.long_memory import (
    LongTermMemoryAction,
    LongTermMemoryOperation,
    MemoryRetrievalMode,
)
from app.services.models import (
    AuditEventRecord,
    CentreRecord,
    ChildRecord,
    ClassRecord,
    ConversationEventRecord,
    ConversationRunRecord,
    ConversationRunResultRecord,
    ConversationSessionRecord,
    EducationalRecord,
    EducationalRecordObservation,
    LongTermMemoryRecord,
    ObservationChildRecord,
    ObservationRecord,
    RecordExport,
    TeacherClassMembershipRecord,
    TeacherRecord,
    ToolActionRequest,
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

SYNTHETIC_CENTRE = {
    "centre_id": "demo-centre",
    "name": "EasyTeaching Demo Centre",
    "state": "NSW",
    "suburb": "Sydney",
    "latitude": -33.8688,
    "longitude": 151.2093,
    "timezone": "Australia/Sydney",
    "active": True,
}


def _contains_pattern(value: str) -> str:
    """Build a literal case-insensitive contains pattern for SQL LIKE."""

    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
SYNTHETIC_TEACHER = {
    "teacher_id": "teacher-001",
    "auth_user_id": None,
    "centre_id": "demo-centre",
    "display_name": "Demo Educator",
    "active": True,
}

ACTIVE_CONVERSATION_RUN_STATUSES = (
    "accepted",
    "running",
    "waiting_for_approval",
)


class AsyncEasyTeachingStore:
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
            if await session.get(CentreRecord, SYNTHETIC_CENTRE["centre_id"]) is None:
                session.add(CentreRecord(**SYNTHETIC_CENTRE))
            if await session.get(TeacherRecord, SYNTHETIC_TEACHER["teacher_id"]) is None:
                session.add(TeacherRecord(**SYNTHETIC_TEACHER))
            for profile in SYNTHETIC_CLASS_PROFILES:
                if await session.get(ClassRecord, profile["class_id"]) is None:
                    session.add(
                        ClassRecord(
                            class_id=profile["class_id"],
                            centre_id=SYNTHETIC_CENTRE["centre_id"],
                            name=profile["name"],
                            age_group=profile["age_group"],
                            current_focus=profile["interests"],
                            active=True,
                        )
                    )
                for child_number in range(1, profile["child_count"] + 1):
                    child_id = f'{profile["class_id"]}-child-{child_number:02d}'
                    if await session.get(ChildRecord, child_id) is None:
                        session.add(
                            ChildRecord(
                                child_id=child_id,
                                class_id=profile["class_id"],
                                display_code=f"Child {child_number:02d}",
                                active=True,
                            )
                        )
                membership_id = f'{SYNTHETIC_TEACHER["teacher_id"]}:{profile["class_id"]}'
                if await session.get(TeacherClassMembershipRecord, membership_id) is None:
                    session.add(
                        TeacherClassMembershipRecord(
                            membership_id=membership_id,
                            teacher_id=SYNTHETIC_TEACHER["teacher_id"],
                            class_id=profile["class_id"],
                            role="educator",
                            active=True,
                        )
                    )
            await session.commit()

    async def close(self) -> None:
        await self.engine.dispose()

    async def get_class_context(
        self,
        *,
        teacher_id: str,
        class_id: str,
        memory_query: Optional[str] = None,
        memory_limit: int = 4,
        include_children: bool = False,
    ) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            centre_id = await self._require_class_access(session, teacher_id, class_id)
            class_record = await session.get(ClassRecord, class_id)
            if class_record is None or not class_record.active:
                return None
            child_count = (
                await session.execute(
                    select(func.count(ChildRecord.child_id)).where(
                        ChildRecord.class_id == class_id,
                        ChildRecord.active.is_(True),
                    )
                )
            ).scalar_one()
            memory_statement = (
                select(LongTermMemoryRecord)
                .where(
                    LongTermMemoryRecord.scope == "class",
                    LongTermMemoryRecord.scope_id == class_id,
                    LongTermMemoryRecord.is_active.is_(True),
                )
                .order_by(
                    LongTermMemoryRecord.importance.desc(),
                    LongTermMemoryRecord.updated_at.desc(),
                )
                .limit(max(memory_limit * 4, memory_limit))
            )
            memories = (await session.execute(memory_statement)).scalars().all()
            children = []
            if include_children:
                child_records = (
                    await session.execute(
                        select(ChildRecord)
                        .where(
                            ChildRecord.class_id == class_id,
                            ChildRecord.active.is_(True),
                        )
                        .order_by(ChildRecord.display_code)
                    )
                ).scalars().all()
                children = [
                    {"child_id": item.child_id, "display_code": item.display_code}
                    for item in child_records
                ]
        tokens = [item.lower() for item in (memory_query or "").split() if item]
        selected = [
            item
            for item in memories
            if not tokens or any(token in item.content.lower() for token in tokens)
        ][:memory_limit]
        return {
            "centre_id": centre_id,
            "class_id": class_record.class_id,
            "name": class_record.name,
            "age_group": class_record.age_group,
            "child_count": int(child_count),
            "current_focus": class_record.current_focus,
            "children": children,
            "class_memories": [self._long_term_memory_to_dict(item) for item in selected],
        }

    async def get_centre_location(
        self,
        *,
        teacher_id: str,
        class_id: str,
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            centre_id = await self._require_class_access(session, teacher_id, class_id)
            centre = await session.get(CentreRecord, centre_id)
            if centre is None or not centre.active:
                raise ValueError("Centre is unavailable")
            return {
                "centre_id": centre.centre_id,
                "state": centre.state,
                "suburb": centre.suburb,
                "timezone": centre.timezone,
                "latitude": centre.latitude,
                "longitude": centre.longitude,
            }

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

    async def get_conversation_artifact(
        self,
        *,
        source_request_id: str,
        session_id: str,
        teacher_id: str,
        class_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read one reusable draft inside its trusted conversation scope."""

        async with self.session_factory() as session:
            conversation = await session.get(ConversationSessionRecord, session_id)
            if conversation is None:
                return None
            if (
                conversation.teacher_id != teacher_id
                or conversation.class_id != class_id
            ):
                raise ValueError("Conversation artifact belongs to another scope")

            result = await session.get(ConversationRunResultRecord, source_request_id)
            if result is None or result.session_id != session_id:
                return None
            draft = result.draft or {}
            approval_status = (result.approval or {}).get("status")
            content = str(draft.get("content") or "").strip()
            if (
                approval_status != "not_required"
                or not bool(draft.get("is_draft", True))
                or not content
            ):
                return None

            saved_action = (
                await session.execute(
                    select(ToolActionRequest)
                    .where(
                        ToolActionRequest.session_id == session_id,
                        ToolActionRequest.status == "executed",
                        ToolActionRequest.tool_name.in_(
                            ("save_observation", "save_educational_record")
                        ),
                        ToolActionRequest.arguments["source_request_id"].as_string()
                        == source_request_id,
                    )
                    .order_by(ToolActionRequest.created_at.desc())
                    .limit(1)
                )
            ).scalars().first()
            return {
                "source_request_id": result.request_id,
                "title": str(draft.get("title") or "").strip() or None,
                "content": content,
                "content_chars": len(content),
                "created_at": result.created_at.isoformat(),
                "status": "saved" if saved_action is not None else "unsaved",
            }

    async def get_conversation_workspace(
        self,
        *,
        session_id: str,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> Dict[str, Any]:
        """Return trusted references to the latest draft and approved record write."""

        async with self.session_factory() as session:
            conversation = await session.get(ConversationSessionRecord, session_id)
            if conversation is None:
                return {}
            if teacher_id is not None and conversation.teacher_id != teacher_id:
                raise ValueError("Conversation workspace belongs to another teacher")
            if class_id is not None and conversation.class_id != class_id:
                raise ValueError("Conversation workspace belongs to another class")

            results = (
                await session.execute(
                    select(ConversationRunResultRecord)
                    .where(ConversationRunResultRecord.session_id == session_id)
                    .order_by(ConversationRunResultRecord.created_at.asc())
                )
            ).scalars().all()
            artifacts = []
            for result in results:
                approval_status = (result.approval or {}).get("status")
                draft = result.draft or {}
                content = str(draft.get("content", "")).strip()
                # Clarification/error messages are conversation turns, not reusable
                # generated artefacts. Historical content itself is not projected
                # back into model context; only the immutable reference metadata is.
                if (
                    approval_status == "not_required"
                    and bool(draft.get("is_draft", True))
                    and content
                ):
                    artifacts.append({
                        "artifact_number": len(artifacts) + 1,
                        "source_request_id": result.request_id,
                        "title": draft.get("title"),
                        "is_draft": True,
                        "content_chars": len(content),
                        "created_at": result.created_at.isoformat(),
                    })
            recent_artifacts = artifacts[-8:]
            for offset, artifact in enumerate(reversed(recent_artifacts)):
                artifact["position_from_latest"] = offset
            current_artifact = artifacts[-1] if artifacts else None

            actions = (
                await session.execute(
                    select(ToolActionRequest)
                    .where(
                        ToolActionRequest.session_id == session_id,
                        ToolActionRequest.status == "executed",
                    )
                    .order_by(ToolActionRequest.created_at.asc())
                )
            ).scalars().all()
            saved_records = []
            for action in actions:
                if action.tool_name not in {
                    "save_observation",
                    "save_educational_record",
                }:
                    continue
                result = action.result or {}
                saved_records.append({
                    "save_number": len(saved_records) + 1,
                    "tool_name": action.tool_name,
                    "source_request_id": (action.arguments or {}).get(
                        "source_request_id"
                    ),
                    "record_id": result.get("record_id")
                    or result.get("observation_id"),
                    "record_type": result.get("record_type"),
                    "title": result.get("title"),
                    "created_at": action.created_at.isoformat(),
                })
            recent_saved_records = saved_records[-8:]
            recent_saved_record = saved_records[-1] if saved_records else None
            saved_source_ids = {
                item["source_request_id"]
                for item in saved_records
                if item.get("source_request_id")
            }
            for artifact in artifacts:
                artifact["status"] = (
                    "saved"
                    if artifact["source_request_id"] in saved_source_ids
                    else "unsaved"
                )
        return {
            "current_artifact": current_artifact,
            "recent_saved_record": recent_saved_record,
            "recent_artifacts": recent_artifacts,
            "recent_saved_records": recent_saved_records,
        }

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

    async def query_records(
        self,
        *,
        teacher_id: str,
        class_id: str,
        record_type: str,
        search_text: Optional[str] = None,
        child_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            await self._require_class_access(session, teacher_id, class_id)
            if child_id is not None:
                child = await session.get(ChildRecord, child_id)
                if child is None or child.class_id != class_id or not child.active:
                    raise ValueError("Child is outside the trusted class scope")

            results: List[Dict[str, Any]] = []
            if record_type in {"observation", "all"}:
                statement = select(ObservationRecord).where(
                    ObservationRecord.class_id == class_id
                )
                if child_id:
                    statement = statement.join(ObservationChildRecord).where(
                        ObservationChildRecord.child_id == child_id
                    )
                if date_from:
                    statement = statement.where(ObservationRecord.observed_at >= date_from)
                if date_to:
                    statement = statement.where(ObservationRecord.observed_at <= date_to)
                if status:
                    statement = statement.where(ObservationRecord.status == status)
                if search_text:
                    pattern = _contains_pattern(search_text)
                    statement = statement.where(
                        or_(
                            ObservationRecord.setting.ilike(pattern, escape="\\"),
                            ObservationRecord.objective_text.ilike(pattern, escape="\\"),
                            ObservationRecord.educator_actions.ilike(pattern, escape="\\"),
                        )
                    )
                observations = (
                    await session.execute(
                        statement.order_by(ObservationRecord.observed_at.desc()).limit(limit)
                    )
                ).scalars().all()
                for item in observations:
                    child_ids = (
                        await session.execute(
                            select(ObservationChildRecord.child_id).where(
                                ObservationChildRecord.observation_id
                                == item.observation_id
                            )
                        )
                    ).scalars().all()
                    results.append(self._observation_to_dict(item, list(child_ids)))

            if record_type in {"educational_record", "all"}:
                statement = select(EducationalRecord).where(
                    EducationalRecord.class_id == class_id
                )
                if child_id:
                    statement = (
                        statement.join(EducationalRecordObservation)
                        .join(
                            ObservationChildRecord,
                            ObservationChildRecord.observation_id
                            == EducationalRecordObservation.observation_id,
                        )
                        .where(ObservationChildRecord.child_id == child_id)
                    )
                if date_from:
                    statement = statement.where(EducationalRecord.created_at >= date_from)
                if date_to:
                    statement = statement.where(EducationalRecord.created_at <= date_to)
                if status:
                    statement = statement.where(EducationalRecord.status == status)
                if search_text:
                    pattern = _contains_pattern(search_text)
                    statement = statement.where(
                        or_(
                            EducationalRecord.title.ilike(pattern, escape="\\"),
                            EducationalRecord.analysis.ilike(pattern, escape="\\"),
                        )
                    )
                records = (
                    await session.execute(
                        statement.order_by(EducationalRecord.created_at.desc()).limit(limit)
                    )
                ).scalars().unique().all()
                for item in records:
                    observation_ids = (
                        await session.execute(
                            select(EducationalRecordObservation.observation_id).where(
                                EducationalRecordObservation.record_id == item.record_id
                            )
                        )
                    ).scalars().all()
                    results.append(
                        self._educational_record_to_dict(item, list(observation_ids))
                    )

        return sorted(results, key=lambda item: item["created_at"], reverse=True)[:limit]

    async def save_observation(
        self,
        *,
        teacher_id: str,
        class_id: str,
        child_ids: List[str],
        observed_at: datetime,
        setting: str,
        objective_text: str,
        educator_actions: Optional[str],
        status: str,
        source_request_id: Optional[str],
        idempotency_key: str,
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            centre_id = await self._require_class_access(session, teacher_id, class_id)
            existing = (
                await session.execute(
                    select(ObservationRecord).where(
                        ObservationRecord.idempotency_key == idempotency_key
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                if (
                    existing.author_teacher_id != teacher_id
                    or existing.class_id != class_id
                ):
                    raise ValueError("Idempotency key belongs to another record scope")
                links = (
                    await session.execute(
                        select(ObservationChildRecord.child_id).where(
                            ObservationChildRecord.observation_id
                            == existing.observation_id
                        )
                    )
                ).scalars().all()
                return self._observation_to_dict(existing, list(links))
            for child_id in child_ids:
                child = await session.get(ChildRecord, child_id)
                if child is None or child.class_id != class_id or not child.active:
                    raise ValueError("Child is outside the trusted class scope")
            record = ObservationRecord(
                observation_id=str(uuid4()),
                centre_id=centre_id,
                class_id=class_id,
                author_teacher_id=teacher_id,
                observed_at=observed_at,
                setting=setting,
                objective_text=objective_text,
                educator_actions=educator_actions,
                status=status,
                source_request_id=source_request_id,
                idempotency_key=idempotency_key,
                version=1,
            )
            session.add(record)
            for child_id in child_ids:
                session.add(
                    ObservationChildRecord(
                        link_id=str(uuid4()),
                        observation_id=record.observation_id,
                        child_id=child_id,
                    )
                )
            session.add(
                self._audit_event(
                    teacher_id=teacher_id,
                    class_id=class_id,
                    action="create",
                    resource_type="observation",
                    resource_id=record.observation_id,
                    tool_name="save_observation",
                )
            )
            await session.commit()
        return self._observation_to_dict(record, child_ids)

    async def save_educational_record(
        self,
        *,
        teacher_id: str,
        class_id: str,
        record_type: str,
        title: str,
        analysis: str,
        curriculum_links: List[Dict[str, Any]],
        next_steps: List[str],
        observation_ids: List[str],
        status: str,
        source_request_id: Optional[str],
        idempotency_key: str,
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            centre_id = await self._require_class_access(session, teacher_id, class_id)
            existing = (
                await session.execute(
                    select(EducationalRecord).where(
                        EducationalRecord.idempotency_key == idempotency_key
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                if (
                    existing.author_teacher_id != teacher_id
                    or existing.class_id != class_id
                ):
                    raise ValueError("Idempotency key belongs to another record scope")
                links = (
                    await session.execute(
                        select(EducationalRecordObservation.observation_id).where(
                            EducationalRecordObservation.record_id == existing.record_id
                        )
                    )
                ).scalars().all()
                return self._educational_record_to_dict(existing, list(links))
            for observation_id in observation_ids:
                observation = await session.get(ObservationRecord, observation_id)
                if observation is None or observation.class_id != class_id:
                    raise ValueError("Observation is outside the trusted class scope")
            now = datetime.utcnow()
            record = EducationalRecord(
                record_id=str(uuid4()),
                centre_id=centre_id,
                class_id=class_id,
                author_teacher_id=teacher_id,
                record_type=record_type,
                title=title,
                analysis=analysis,
                curriculum_links=curriculum_links,
                next_steps=next_steps,
                status=status,
                source_request_id=source_request_id,
                idempotency_key=idempotency_key,
                version=1,
                approved_by=teacher_id if status == "final" else None,
                approved_at=now if status == "final" else None,
            )
            session.add(record)
            for observation_id in observation_ids:
                session.add(
                    EducationalRecordObservation(
                        link_id=str(uuid4()),
                        record_id=record.record_id,
                        observation_id=observation_id,
                    )
                )
            session.add(
                self._audit_event(
                    teacher_id=teacher_id,
                    class_id=class_id,
                    action="create",
                    resource_type="educational_record",
                    resource_id=record.record_id,
                    tool_name="save_educational_record",
                )
            )
            await session.commit()
        return self._educational_record_to_dict(record, observation_ids)

    async def get_exportable_records(
        self,
        *,
        teacher_id: str,
        class_id: str,
        record_ids: List[str],
    ) -> List[Dict[str, Any]]:
        async with self.session_factory() as session:
            await self._require_class_access(session, teacher_id, class_id)
            results: List[Dict[str, Any]] = []
            for record_id in dict.fromkeys(record_ids):
                observation = await session.get(ObservationRecord, record_id)
                if observation is not None:
                    if observation.class_id != class_id:
                        raise ValueError("Record is outside the trusted class scope")
                    child_ids = (
                        await session.execute(
                            select(ObservationChildRecord.child_id).where(
                                ObservationChildRecord.observation_id == record_id
                            )
                        )
                    ).scalars().all()
                    results.append(self._observation_to_dict(observation, list(child_ids)))
                    continue
                educational = await session.get(EducationalRecord, record_id)
                if educational is None:
                    raise ValueError(f"Record does not exist: {record_id}")
                if educational.class_id != class_id:
                    raise ValueError("Record is outside the trusted class scope")
                observation_ids = (
                    await session.execute(
                        select(EducationalRecordObservation.observation_id).where(
                            EducationalRecordObservation.record_id == record_id
                        )
                    )
                ).scalars().all()
                results.append(
                    self._educational_record_to_dict(educational, list(observation_ids))
                )
        return results

    async def save_record_export(
        self,
        *,
        teacher_id: str,
        class_id: str,
        record_ids: List[str],
        format: str,
        template_name: str,
        storage_path: str,
        checksum: str,
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            await self._require_class_access(session, teacher_id, class_id)
            record = RecordExport(
                export_id=str(uuid4()),
                teacher_id=teacher_id,
                record_ids=record_ids,
                format=format,
                template_name=template_name,
                storage_path=storage_path,
                checksum=checksum,
                status="ready",
                expires_at=datetime.utcnow() + timedelta(days=7),
            )
            session.add(record)
            session.add(
                self._audit_event(
                    teacher_id=teacher_id,
                    class_id=class_id,
                    action="export",
                    resource_type="record_export",
                    resource_id=record.export_id,
                    tool_name="export_records",
                )
            )
            await session.commit()
        return {
            "export_id": record.export_id,
            "format": record.format,
            "template_name": record.template_name,
            "storage_path": record.storage_path,
            "checksum": record.checksum,
            "status": record.status,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        }

    async def get_record_export(
        self,
        *,
        teacher_id: str,
        class_id: str,
        export_id: str,
    ) -> Dict[str, Any]:
        """Return only an export owned by the trusted teacher/class scope."""

        async with self.session_factory() as session:
            await self._require_class_access(session, teacher_id, class_id)
            record = await session.get(RecordExport, export_id)
            if record is None or record.teacher_id != teacher_id:
                raise ValueError("Record export does not exist in the trusted scope")
            # The export table predates centre scoping. Revalidate every source
            # record against the current class before releasing its local path.
            await self.get_exportable_records(
                teacher_id=teacher_id,
                class_id=class_id,
                record_ids=list(record.record_ids),
            )
            if record.expires_at and record.expires_at < datetime.utcnow():
                raise ValueError("Record export has expired")
            return {
                "export_id": record.export_id,
                "format": record.format,
                "storage_path": record.storage_path,
                "checksum": record.checksum,
                "status": record.status,
            }

    async def create_tool_action_request(
        self,
        *,
        request_id: str,
        session_id: str,
        teacher_id: str,
        class_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        preview: Dict[str, Any],
        ttl_minutes: int = 30,
    ) -> Dict[str, Any]:
        serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        arguments_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        action = ToolActionRequest(
            action_id=str(uuid4()),
            request_id=request_id,
            session_id=session_id,
            teacher_id=teacher_id,
            class_id=class_id,
            tool_name=tool_name,
            arguments=arguments,
            arguments_hash=arguments_hash,
            preview=preview,
            status="pending",
            expires_at=datetime.utcnow() + timedelta(minutes=ttl_minutes),
        )
        async with self.session_factory() as session:
            existing = (
                await session.execute(
                    select(ToolActionRequest).where(
                        ToolActionRequest.request_id == request_id
                    )
                )
            ).scalar_one_or_none()
            if existing is not None:
                if (
                    existing.session_id != session_id
                    or existing.teacher_id != teacher_id
                    or existing.class_id != class_id
                    or existing.tool_name != tool_name
                    or existing.arguments_hash != arguments_hash
                ):
                    raise ValueError("Request already owns a different frozen action")
                return self._tool_action_to_dict(existing)
            session.add(action)
            await session.commit()
        return self._tool_action_to_dict(action)

    async def claim_tool_action_request(self, action_id: str) -> Dict[str, Any]:
        """Atomically reserve one approved action before its side effect runs."""

        now = datetime.utcnow()
        async with self.session_factory() as session:
            claimed = (
                await session.execute(
                    update(ToolActionRequest)
                    .where(
                        ToolActionRequest.action_id == action_id,
                        ToolActionRequest.status == "pending",
                        ToolActionRequest.expires_at >= now,
                    )
                    .values(status="executing", decided_at=now)
                    .returning(ToolActionRequest)
                )
            ).scalar_one_or_none()
            if claimed is not None:
                await session.commit()
                return self._tool_action_to_dict(claimed)
            action = await session.get(ToolActionRequest, action_id)
            if action is None:
                raise ValueError("Tool action request does not exist")
            if action.status == "pending" and action.expires_at < now:
                action.status = "expired"
                action.decided_at = now
                await session.commit()
                raise ValueError("Tool action request has expired")
            raise ValueError("Tool action request has already been decided")

    async def get_tool_action_request(self, action_id: str) -> Optional[Dict[str, Any]]:
        async with self.session_factory() as session:
            action = await session.get(ToolActionRequest, action_id)
            return None if action is None else self._tool_action_to_dict(action)

    async def finish_tool_action_request(
        self,
        action_id: str,
        *,
        status: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with self.session_factory() as session:
            action = await session.get(ToolActionRequest, action_id)
            if action is None:
                raise ValueError("Tool action request does not exist")
            expected_status = "executing" if status in {"executed", "failed"} else "pending"
            if action.status != expected_status:
                raise ValueError("Tool action request has already been decided")
            if expected_status == "pending" and action.expires_at < datetime.utcnow():
                action.status = "expired"
                await session.commit()
                raise ValueError("Tool action request has expired")
            action.status = status
            action.result = result
            action.decided_at = datetime.utcnow()
            action.executed_at = datetime.utcnow() if status == "executed" else None
            session.add(
                self._audit_event(
                    teacher_id=action.teacher_id,
                    class_id=action.class_id,
                    action=f"approval_{status}",
                    resource_type="tool_action",
                    resource_id=action.action_id,
                    tool_name=action.tool_name,
                    result="success" if status in {"executed", "rejected"} else "failed",
                )
            )
            await session.commit()
        return self._tool_action_to_dict(action)

    async def _require_class_access(
        self,
        session: AsyncSession,
        teacher_id: str,
        class_id: str,
    ) -> str:
        membership = (
            await session.execute(
                select(TeacherClassMembershipRecord)
                .where(
                    TeacherClassMembershipRecord.teacher_id == teacher_id,
                    TeacherClassMembershipRecord.class_id == class_id,
                    TeacherClassMembershipRecord.active.is_(True),
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        if membership is None:
            raise ValueError("Teacher is not authorised for the trusted class scope")
        class_record = await session.get(ClassRecord, class_id)
        if class_record is None or not class_record.active:
            raise ValueError("Class is unavailable")
        return class_record.centre_id

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

    @staticmethod
    def _observation_to_dict(
        record: ObservationRecord,
        child_ids: List[str],
    ) -> Dict[str, Any]:
        return {
            "record_type": "observation",
            "observation_id": record.observation_id,
            "centre_id": record.centre_id,
            "class_id": record.class_id,
            "author_teacher_id": record.author_teacher_id,
            "child_ids": child_ids,
            "observed_at": record.observed_at.isoformat(),
            "setting": record.setting,
            "objective_text": record.objective_text,
            "educator_actions": record.educator_actions,
            "status": record.status,
            "version": record.version,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    @staticmethod
    def _educational_record_to_dict(
        record: EducationalRecord,
        observation_ids: List[str],
    ) -> Dict[str, Any]:
        return {
            "record_type": "educational_record",
            "educational_record_type": record.record_type,
            "record_id": record.record_id,
            "centre_id": record.centre_id,
            "class_id": record.class_id,
            "author_teacher_id": record.author_teacher_id,
            "title": record.title,
            "analysis": record.analysis,
            "curriculum_links": record.curriculum_links,
            "next_steps": record.next_steps,
            "observation_ids": observation_ids,
            "status": record.status,
            "version": record.version,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    @staticmethod
    def _tool_action_to_dict(action: ToolActionRequest) -> Dict[str, Any]:
        return {
            "action_id": action.action_id,
            "request_id": action.request_id,
            "session_id": action.session_id,
            "teacher_id": action.teacher_id,
            "class_id": action.class_id,
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "arguments_hash": action.arguments_hash,
            "preview": action.preview,
            "status": action.status,
            "result": action.result,
            "expires_at": action.expires_at.isoformat(),
            "created_at": action.created_at.isoformat(),
        }

    @staticmethod
    def _audit_event(
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        action: str,
        resource_type: str,
        resource_id: Optional[str],
        tool_name: Optional[str],
        result: str = "success",
    ) -> AuditEventRecord:
        return AuditEventRecord(
            audit_id=str(uuid4()),
            teacher_id=teacher_id,
            class_id=class_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            tool_name=tool_name,
            result=result,
            metadata_json={},
        )
