import asyncio
from unittest.mock import AsyncMock

from langgraph.checkpoint.memory import MemorySaver

from app.schemas import CapabilityCall, MainDecision, GraphState, ObservationStatus
from app.tools.dynamic_drive import configure_lazy_drive, tools_for_run, register_dynamic_drive
from app.tools.registry import ToolRegistry
from app.workflows.main_react_graph import build_main_react_graph
from tests.test_dynamic_drive_registration import Client
from tests.test_main_react_graph import NoMemoryStore, NoMemoryExtractor, StubWorkerRunner


def configure(registry, client):
    configure_lazy_drive(registry, None, client=client, user_google_email="owner@example.com")


def test_configuration_has_no_io_and_concurrent_load_is_cached():
    async def scenario():
        registry, client = ToolRegistry(), Client()
        client.list_tools = AsyncMock(wraps=client.list_tools)
        configure(registry, client)
        client.list_tools.assert_not_called()
        assert [tool.name for tool in registry.list_tools()] == ["load_drive_tools"]
        first, second = await asyncio.gather(
            registry.execute_async("load_drive_tools", {}),
            registry.execute_async("load_drive_tools", {}))
        assert first.success and second.success
        assert first.data["tool_names"] == second.data["tool_names"]
        assert client.list_tools.await_count == 1
        assert not client.calls
        # Another run cannot see the cached dynamic definitions automatically.
        assert [t.name for t in await tools_for_run(registry, {})] == ["load_drive_tools"]
    asyncio.run(scenario())


def test_failed_discovery_can_retry_without_partial_registration():
    async def scenario():
        registry, client = ToolRegistry(), Client()
        client.list_tools = AsyncMock(side_effect=[ConnectionError("offline"), client.tools])
        configure(registry, client)
        assert not (await registry.execute_async("load_drive_tools", {})).success
        assert [t.name for t in registry.list_tools()] == ["load_drive_tools"]
        assert (await registry.execute_async("load_drive_tools", {})).success
    asyncio.run(scenario())


def test_resumed_call_reconstructs_catalog_but_changed_schema_fails_closed():
    async def scenario():
        client = Client()
        original = ToolRegistry()
        configure(original, client)
        loaded = await original.execute_async("load_drive_tools", {})
        name = loaded.data["tool_names"][0]
        resumed = ToolRegistry()
        configure(resumed, client)
        result = await resumed.execute_async(name, {"query": "abc"},
            approved=True, allowed_tool_names=[name])
        assert result.success and len(client.calls) == 1
        client.tools[0].input_schema["properties"]["query"]["minLength"] = 10
        changed = ToolRegistry()
        configure(changed, client)
        assert not (await changed.execute_async(name, {"query": "abc"}, approved=True)).success
        assert len(client.calls) == 1
    asyncio.run(scenario())


def test_version_refresh_does_not_remove_active_definitions():
    async def scenario():
        registry, client = ToolRegistry(), Client()
        old = (await register_dynamic_drive(registry, None, client=client,
            user_google_email="owner@example.com"))[0]
        client.tools[0].input_schema["properties"]["query"]["minLength"] = 10
        new = (await register_dynamic_drive(registry, None, client=client,
            user_google_email="owner@example.com"))[0]
        assert old.name != new.name
        assert registry.get(old.name) is not None
        assert registry.get(new.name) is not None
        assert not (await registry.execute_async(old.name, {"query": "abc"})).success
    asyncio.run(scenario())


def test_graph_load_then_next_round_executes_dynamic_tool_and_new_run_is_isolated():
    class Agent:
        calls = 0
        async def decide(self, **kwargs):
            names = [tool.name for tool in kwargs["available_tools"]]
            self.calls += 1
            if self.calls == 1:
                assert "load_drive_tools" in names
                assert not any(name.startswith("drive__") for name in names)
                return MainDecision(reason="Load Drive", tool_calls=[CapabilityCall(
                    name="load_drive_tools", arguments={}, result_key="drive_catalog")])
            if self.calls == 2:
                assert "load_drive_tools" not in names
                dynamic = next(name for name in names if name.startswith("drive__"))
                return MainDecision(reason="Search", tool_calls=[CapabilityCall(
                    name=dynamic, arguments={"query": "abc"}, result_key="search_result")])
            if self.calls == 4:
                assert "load_drive_tools" in names
                assert not any(name.startswith("drive__") for name in names)
            return MainDecision(reason="Done", final_answer="Synthetic draft.")

    async def scenario():
        registry, client = ToolRegistry(), Client()
        configure(registry, client)
        graph = build_main_react_graph(main_agent=Agent(), registry=registry,
            worker_runner=StubWorkerRunner(), long_memory_store=NoMemoryStore(),
            long_memory_extractor=NoMemoryExtractor(), checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "lazy-drive"}}
        first = GraphState.model_validate(await graph.ainvoke(dict(request_id="first",
            session_id="session", user_message="Find an activity draft in Drive"), cfg))
        assert first.observations["search_result"].status is ObservationStatus.COMPLETED
        assert first.tool_call_count == 2
        assert len(client.calls) == 1
        second = GraphState.model_validate(await graph.ainvoke(dict(request_id="second",
            session_id="session", user_message="Create a simple activity draft"), cfg))
        assert second.tool_call_count == 0
        assert len(client.calls) == 1
    asyncio.run(scenario())
