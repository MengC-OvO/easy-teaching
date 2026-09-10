"""Real graph/checkpointer with synthetic tools, no external model/service."""
import asyncio

from langgraph.checkpoint.memory import MemorySaver

from app.schemas import CapabilityCall, GraphState, MainDecision, RiskLevel
from app.services.observation_results import encoded
from app.tools import ToolResult
from app.workflows.main_react_graph import build_main_react_graph
from tests.test_main_react_graph import (
    _registry, SequenceMainAgent, NoMemoryStore, NoMemoryExtractor, StubWorkerRunner,
)
from tests.test_observation_management import Snapshots


def test_large_result_stays_out_of_every_checkpoint_and_next_run_resets():
    class Store(Snapshots, NoMemoryStore):
        pass
    store = Store()
    registry = _registry()
    original = "UNIQUE_ORIGINAL_FACT_" * 4000
    registry.get("tool_a").handler = lambda args: ToolResult.ok(
        data={"value": original}, risk_level=RiskLevel.L0_READ_ONLY)
    agent = SequenceMainAgent([
        MainDecision(reason="Read", tool_calls=[CapabilityCall(
            name="tool_a", arguments={"text": "small query"}, result_key="large")]),
        MainDecision(reason="Done", final_answer="Synthetic draft."),
        MainDecision(reason="New run", final_answer="Second draft."),
    ])
    graph = build_main_react_graph(main_agent=agent, registry=registry,
        worker_runner=StubWorkerRunner(), long_memory_store=store,
        long_memory_extractor=NoMemoryExtractor(), checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "result-test-thread"}}

    async def scenario():
        first = GraphState.model_validate(await graph.ainvoke(dict(
            request_id="req-first", session_id="session", teacher_id="teacher",
            class_id="class", user_message="Create an early childhood activity draft."), cfg))
        assert first.observations["large"].body_ref
        assert first.tool_call_count == 1
        assert len(store.rows) == 1
        assert next(iter(store.rows.values()))["body"]["value"] == original
        async for checkpoint in graph.aget_state_history(cfg):
            assert original[:20000] not in encoded(checkpoint.values)
        second = GraphState.model_validate(await graph.ainvoke(dict(
            request_id="req-second", session_id="session", teacher_id="teacher",
            class_id="class", user_message="Make the draft shorter."), cfg))
        assert not second.observations
        assert not second.pending_observations
        assert second.tool_call_count == 0
    asyncio.run(scenario())
