import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.checkpoint_config import checkpoint_config
from app.schemas import (
    CapabilityCall,
    GraphState,
    LongTermMemoryAction,
    LongTermMemoryCandidate,
    LongTermMemoryOperation,
    LongTermMemoryScope,
    LongTermMemoryType,
    MainDecision,
    MemoryRetrievalMode,
)
from app.agents import WorkerRegistry
from app.services.conversation_workspace import load_workspace, sync_workspace_checkpoint
from app.services.async_store import AsyncEasyTeachingStore
from app.services.models import (
    Base,
    ConversationRunResultRecord,
    ConversationSessionRecord,
    LongTermMemoryRecord,
    ToolActionRequest,
)
from app.workflows import build_main_react_graph
from tests.test_outbox_leases import AsyncSessionShim
from tests.test_main_react_graph import (
    NoMemoryStore, NoMemoryExtractor, SequenceMainAgent, StubWorkerRunner, _registry,
)


class CatalogueStore(NoMemoryStore):
    def __init__(self):
        self.reads = 0
        self.catalogue = {"recent_artifacts": [{
            "source_request_id": "old-draft", "title": "Previous draft",
            "artifact_number": 1, "position_from_latest": 0,
            "content": "Full text must not enter the catalogue",
        }]}

    async def get_conversation_workspace(self, **kwargs):
        self.reads += 1
        return self.catalogue

    async def get_conversation_session(self, session_id):
        return {"thread_id": "thread-1", "teacher_id": None, "class_id": None}


def test_catalogue_loaded_once_per_request_and_updated_without_model_execution():
    async def scenario():
        store = CatalogueStore()
        agent = SequenceMainAgent([
            MainDecision(reason="Inspect", tool_calls=[CapabilityCall(
                name="tool_a", arguments={"text": "hello"}, result_key="a")]),
            MainDecision(reason="Done", final_answer="Hello teacher."),
            MainDecision(reason="Done", final_answer="Next answer."),
        ])
        graph = build_main_react_graph(
            main_agent=agent, registry=_registry(), worker_registry=WorkerRegistry([]),
            worker_runner=StubWorkerRunner(), long_memory_store=store,
            long_memory_extractor=NoMemoryExtractor(), checkpointer=MemorySaver(),
        )
        config = checkpoint_config("thread-1")
        await graph.ainvoke(GraphState(
            request_id="request-1", session_id="session-1", thread_id="thread-1",
            user_message="Hello teacher",
        ), config)
        assert len(agent.calls) == 2
        assert store.reads == 1
        for call in agent.calls:
            assert "old-draft" in call["conversation_context"]
            assert "Full text" not in call["conversation_context"]

        # Simulate committed draft / approval data; the same helper serves both.
        store.catalogue["recent_artifacts"][0]["status"] = "saved"
        store.catalogue["recent_saved_records"] = [{
            "record_id": "record-1", "source_request_id": "old-draft",
        }]
        await sync_workspace_checkpoint(SimpleNamespace(graph=graph, store=store),
                                        session_id="session-1", request_id="request-1")
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ()
        assert len(agent.calls) == 2
        assert snapshot.values["workspace"]["recent_artifacts"][0]["status"] == "saved"
        assert snapshot.values["workspace"]["recent_saved_records"][0]["record_id"] == "record-1"
        assert "content" not in snapshot.values["workspace"]["recent_artifacts"][0]

        # A new request reconciles the DB, including clearing stale entries.
        store.catalogue = {}
        await graph.ainvoke({"request_id": "request-2", "user_message": "Hello again"}, config)
        assert store.reads == 3  # initialize + post-commit + next initialize
        assert "old-draft" not in agent.calls[-1]["conversation_context"]
        assert (await graph.aget_state(config)).values["workspace"]["recent_artifacts"] == []
    asyncio.run(scenario())


@pytest.mark.parametrize("next_nodes,request_id", [(('main_react',), 'request-1'), ((), 'other')])
def test_sync_does_not_overwrite_pending_or_different_run(next_nodes, request_id):
    async def scenario():
        graph = SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(
                values={"request_id": request_id, "session_id": "session-1"}, next=next_nodes)),
            aupdate_state=AsyncMock(),
        )
        store = CatalogueStore()
        await sync_workspace_checkpoint(SimpleNamespace(graph=graph, store=store),
                                        session_id="session-1", request_id="request-1")
        graph.aupdate_state.assert_not_called()
        assert store.reads == 0
    asyncio.run(scenario())


def test_cache_failure_is_repaired_by_next_load():
    async def scenario():
        graph = SimpleNamespace(
            aget_state=AsyncMock(return_value=SimpleNamespace(
                values={"request_id": "request-1", "session_id": "session-1"}, next=())),
            aupdate_state=AsyncMock(side_effect=RuntimeError("checkpoint unavailable")),
        )
        store = CatalogueStore()
        await sync_workspace_checkpoint(SimpleNamespace(graph=graph, store=store),
                                        session_id="session-1", request_id="request-1")
        workspace = await load_workspace(store, session_id="session-1", teacher_id=None, class_id=None)
        assert workspace["recent_artifacts"][0]["source_request_id"] == "old-draft"
    asyncio.run(scenario())


def test_database_workspace_reads_only_latest_eight_and_preserves_global_numbers():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    base = datetime(2026, 1, 1)
    with Session(engine) as db:
        db.add(ConversationSessionRecord(
            session_id="session-1", thread_id="thread-1",
            teacher_id="teacher-1", class_id="class-1",
        ))
        for index in range(12):
            source_request_id = f"draft-{index + 1}"
            db.add(ConversationRunResultRecord(
                request_id=source_request_id,
                session_id="session-1",
                draft={"title": source_request_id, "content": "draft", "is_draft": True},
                approval={"status": "not_required"},
                citations=[],
                created_at=base + timedelta(minutes=index),
            ))
            arguments = {"source_request_id": source_request_id}
            db.add(ToolActionRequest(
                action_id=f"action-{index + 1}", request_id=f"save-{index + 1}",
                session_id="session-1", teacher_id="teacher-1", class_id="class-1",
                tool_name="save_observation", arguments=arguments,
                arguments_hash="unused", preview={}, status="executed",
                result={"record_id": f"record-{index + 1}"},
                expires_at=base + timedelta(days=1),
                created_at=base + timedelta(minutes=index),
            ))
        db.commit()

    store = object.__new__(AsyncEasyTeachingStore)
    store.session_factory = lambda: AsyncSessionShim(engine)
    try:
        workspace = asyncio.run(store.get_conversation_workspace(
            session_id="session-1", teacher_id="teacher-1", class_id="class-1",
        ))
    finally:
        engine.dispose()

    assert [item["source_request_id"] for item in workspace["recent_artifacts"]] == [
        f"draft-{index}" for index in range(5, 13)
    ]
    assert [item["artifact_number"] for item in workspace["recent_artifacts"]] == list(range(5, 13))
    assert workspace["current_artifact"]["source_request_id"] == "draft-12"
    assert workspace["current_artifact"]["position_from_latest"] == 0
    assert workspace["current_artifact"]["status"] == "saved"
    assert [item["save_number"] for item in workspace["recent_saved_records"]] == list(range(5, 13))


class FakeReadCache:
    def __init__(self):
        self.values = {}
        self.gets = 0
        self.sets = 0
        self.deletes = 0

    async def get(self, key):
        self.gets += 1
        return self.values.get(key)

    async def set(self, key, value, *, ex):
        assert ex == 300
        self.sets += 1
        self.values[key] = value

    async def delete(self, key):
        self.deletes += 1
        self.values.pop(key, None)


def test_teacher_profile_memories_use_scoped_redis_cache():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(LongTermMemoryRecord(
            memory_id="memory-1", scope="teacher", scope_id="teacher-1",
            memory_type="teacher_preference", content="Prefers concise plans.",
            reason="Explicit preference", retrieval_mode="profile",
            importance=3, is_active=True,
        ))
        db.commit()

    sessions_opened = 0

    def session_factory():
        nonlocal sessions_opened
        sessions_opened += 1
        return AsyncSessionShim(engine)

    store = object.__new__(AsyncEasyTeachingStore)
    store.session_factory = session_factory
    cache = FakeReadCache()
    store.configure_read_cache(cache, ttl_seconds=300)
    try:
        first = asyncio.run(store.list_profile_memories(teacher_id="teacher-1", limit=4))
        second = asyncio.run(store.list_profile_memories(teacher_id="teacher-1", limit=4))
        operation = LongTermMemoryOperation(
            action=LongTermMemoryAction.UPDATE,
            memory_id="memory-1",
            candidate=LongTermMemoryCandidate(
                scope=LongTermMemoryScope.TEACHER,
                scope_id="teacher-1",
                memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
                content="Prefers detailed plans.",
                reason="Updated preference",
                retrieval_mode=MemoryRetrievalMode.PROFILE,
                importance=3,
            ),
            reason="Teacher changed the preference",
        )
        asyncio.run(store.apply_long_term_memory_operation(
            operation, teacher_id="teacher-1", class_id="class-1",
        ))
        refreshed = asyncio.run(store.list_profile_memories(
            teacher_id="teacher-1", limit=4,
        ))
    finally:
        engine.dispose()

    assert first == second
    assert first[0]["content"] == "Prefers concise plans."
    assert refreshed[0]["content"] == "Prefers detailed plans."
    assert sessions_opened == 3  # first read, update, post-invalidation read
    assert cache.sets == 2 and cache.gets == 3 and cache.deletes == 1
    assert "teacher-1" not in next(iter(cache.values))
