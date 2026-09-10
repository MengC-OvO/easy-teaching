import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas import CapabilityObservation, GraphState, ObservationStatus
from app.schemas.observation_updates import merge_observation_updates, replace_observations
from app.services.observation_results import ObservationResults, ResultSummary, encoded
from app.tools.read_observation import build_read_observation_tool, ReadObservationInput
from app.tools.definition import ToolExecutionContext
from app.workflows.main_react_graph import observation_delta, merge_observations


class Snapshots:
    def __init__(self):
        self.rows = {}
        self.reads = 0

    async def save_tool_result_snapshot(self, **row):
        old = self.rows.get(row["body_ref"])
        if old is not None and old != row:
            raise ValueError("conflict")
        self.rows[row["body_ref"]] = copy.deepcopy(row)
        return row["body_ref"]

    async def read_tool_result_snapshot(self, **scope):
        self.reads += 1
        row = self.rows[scope["body_ref"]]
        if any(row[key] != scope[key] for key in ("session_id", "teacher_id", "class_id")):
            raise PermissionError("scope")
        return copy.deepcopy(row)


def config(**overrides):
    return SimpleNamespace(**{
        "observation_inline_chars": 1000, "observation_snapshot_chars": 100000,
        "observation_high_chars": 20000, "observation_low_chars": 10000,
        "observation_summary_enabled": False, **overrides,
    })


def state():
    return GraphState(request_id="request", session_id="session", teacher_id="teacher",
                      class_id="class", user_message="synthetic input")


def observation(key="one", text="small", step=0):
    item = CapabilityObservation(result_key=key, capability_name="text_tool", source_kind="tool",
                                 status="completed", data={"content": text, "record_id": "record-1"})
    return observation_delta([item], request_id="request", step=step)[key]


def test_reducer_parallel_delta_order_replay_conflict_and_explicit_reset():
    first, second = observation("a", step=0), observation("b", step=1)
    merged = merge_observation_updates({"b": second}, {"a": first})
    assert list(merged) == ["a", "b"]
    assert merge_observation_updates(merged, {"a": first}) == merged
    assert merge_observation_updates(merged, {}) == merged
    assert merge_observation_updates(merged, replace_observations({})) == {}
    with pytest.raises(ValueError, match="Conflicting"):
        merge_observation_updates(merged, {"a": observation("a", "different")})


def test_organizing_twice_does_not_repeat_counts_or_trace():
    current = state().model_copy(update={"observations": {"one": observation()}})
    first = merge_observations(current)
    current = current.model_copy(update={key: value for key, value in first.items() if key != "observations"})
    second = merge_observations(current)
    assert "react_step" not in second
    assert "trace" not in second


def test_legacy_pending_only_migrates_unprocessed_results():
    old, new = observation("old"), observation("new", step=1)
    current = state().model_copy(update={"observations": {"old": old},
        "pending_observations": [old, new], "merged_observation_count": 1})
    updated = merge_observations(current)
    assert updated["pending_observations"] == []
    assert updated["processed_observation_keys"] == ["new", "old"]
    assert len(updated["trace"][0].metadata["observations"]) == 1


def test_small_result_never_writes_or_summarizes():
    store = Snapshots()
    provider = SimpleNamespace(generate_structured=AsyncMock())
    manager = ObservationResults(store, provider=provider, config=config())
    result = asyncio.run(manager.prepare(observation(), state()))
    assert result.body_ref is None and result.data["content"] == "small"
    assert store.rows == {}
    provider.generate_structured.assert_not_called()


def test_partial_source_metadata_and_lost_lease_are_preserved():
    from app.services.task_lease import LeaseLostError
    async def scenario():
        store = Snapshots()
        manager = ObservationResults(store, config=config())
        partial = observation().model_copy(update={"data": {"has_more": True, "records": []}})
        assert (await manager.prepare(partial, state())).is_partial
        store.save_tool_result_snapshot = AsyncMock(side_effect=LeaseLostError("stale"))
        with pytest.raises(LeaseLostError):
            await manager.prepare(observation(text="x" * 5000), state())
    asyncio.run(scenario())


def test_large_result_has_immutable_scoped_original_and_idempotent_save():
    async def scenario():
        store = Snapshots()
        manager = ObservationResults(store, config=config())
        original = observation(text="original sentence " * 400)
        archived = await manager.prepare(original, state())
        assert archived.body_ref and archived.is_partial
        assert archived.original_size > len(encoded(archived.data))
        assert (await manager.prepare(original, state())).body_ref == archived.body_ref
        assert len(store.rows) == 1
        assert await manager.resolve_full_content(archived, state()) == original.data
        with pytest.raises(PermissionError):
            await manager.resolve_full_content(archived, state().model_copy(update={"teacher_id": "other"}))
        store.rows[archived.body_ref]["body"]["content"] = "tampered"
        with pytest.raises(ValueError, match="hash"):
            await manager.resolve_full_content(archived, state())
    asyncio.run(scenario())


def test_storage_failure_returns_explicit_failure_without_dangling_ref():
    store = SimpleNamespace(save_tool_result_snapshot=AsyncMock(side_effect=RuntimeError("offline")))
    result = asyncio.run(ObservationResults(store, config=config()).prepare(observation(text="x" * 2000), state()))
    assert result.status is ObservationStatus.FAILED
    assert result.body_ref is None and result.data == {} and result.is_partial


def test_oversized_result_is_rejected_before_snapshot_write():
    store = Snapshots()
    result = asyncio.run(ObservationResults(store, config=config(observation_snapshot_chars=1500)).prepare(observation(text="x" * 2000), state()))
    assert result.status is ObservationStatus.FAILED and not store.rows


def test_read_fragments_progress_and_do_not_create_new_snapshots():
    async def scenario():
        store = Snapshots()
        manager = ObservationResults(store, config=config())
        original = observation(text="0123456789" * 300)
        archived = await manager.prepare(original, state())
        tool = build_read_observation_tool(store)
        context = ToolExecutionContext(**state().model_dump(include={"request_id", "session_id", "teacher_id", "class_id"}))
        parts, cursor = [], 0
        while True:
            result = await tool.async_runtime_handler(ReadObservationInput(
                body_ref=archived.body_ref, section="content", cursor=cursor, limit=700,
            ), context)
            parts.append(result.data["text"])
            if not result.data["has_more"]:
                break
            assert result.data["next_cursor"] > cursor
            cursor = result.data["next_cursor"]
        assert "".join(parts) == original.data["content"]
        assert len(store.rows) == 1
        with pytest.raises(ValueError):
            await tool.async_runtime_handler(ReadObservationInput(body_ref=archived.body_ref, cursor=100000), context)
    asyncio.run(scenario())


def test_budget_archives_old_body_but_preserves_latest_and_reuses_summary():
    async def scenario():
        store = Snapshots()
        provider = SimpleNamespace(generate_structured=AsyncMock(return_value=SimpleNamespace(
            structured=ResultSummary(result_key="old", summary="Old facts."))))
        manager = ObservationResults(store, provider=provider, config=config(
            observation_inline_chars=10000, observation_high_chars=8000,
            observation_low_chars=5500, observation_summary_enabled=True))
        old, new = observation("old", "old sentence " * 400), observation("new", "latest sentence " * 250, step=1)
        organized = await manager.organize({"old": old, "new": new}, state())
        assert organized["old"].body_ref
        assert organized["old"].summary == "Derived summary: Old facts."
        assert organized["new"].data == new.data
        await manager.organize({"old": old, "new": new}, state())
        provider.generate_structured.assert_awaited_once()
    asyncio.run(scenario())


def test_summary_failure_keeps_original_recoverable():
    async def scenario():
        store = Snapshots()
        provider = SimpleNamespace(generate_structured=AsyncMock(side_effect=RuntimeError("offline")))
        manager = ObservationResults(store, provider=provider, config=config(observation_summary_enabled=True))
        original = observation(text="sentence " * 400)
        archived = await manager.archive(original, state())
        result = await manager.summarize(original, archived)
        assert result.summary is None
        assert await manager.resolve_full_content(result, state()) == original.data
    asyncio.run(scenario())


def test_business_view_restores_exact_body_without_mutating_state():
    async def scenario():
        manager = ObservationResults(Snapshots(), config=config())
        original = observation(text="checked candidate " * 300)
        archived = await manager.prepare(original, state())
        current = state().model_copy(update={"observations": {"one": archived}})
        business = await manager.business_state(current)
        assert business.observations["one"].data == original.data
        assert current.observations["one"].data != original.data
    asyncio.run(scenario())
