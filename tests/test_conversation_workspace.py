import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.api.checkpoint_config import checkpoint_config
from app.schemas import GraphState, MainDecision, CapabilityCall
from app.agents import WorkerRegistry
from app.services.conversation_workspace import load_workspace, sync_workspace_checkpoint
from app.workflows import build_main_react_graph
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
