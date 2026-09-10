"""Regression tests for checkpoint ownership, resume, and finalization."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api import execution, recovery
from app.schemas import GraphState, RunStatus
from app.tasks import worker


def snapshot(*, request_id="request-1", session_id="session-1", status="completed", next_nodes=()):
    state = GraphState(
        request_id=request_id, session_id=session_id, user_message="synthetic input",
        workflow_status=status,
    )
    return SimpleNamespace(values=state.model_dump(mode="json"), next=next_nodes)


def runtime_for(saved, *, run_status="running", attempts=2):
    run = {"request_id": "request-1", "session_id": "session-1", "status": run_status}
    task = {
        "execution_attempts": attempts,
        "lease_token": "execution-token",
        "payload": {
            "session_id": "session-1", "thread_id": "thread-1", "message": "new input",
        },
    }
    store = SimpleNamespace(
        claim_conversation_task_for_execution=AsyncMock(return_value=task),
        get_conversation_task=AsyncMock(return_value=task),
        get_conversation_run=AsyncMock(return_value=run),
        finish_conversation_task_execution=AsyncMock(),
        get_conversation_session=AsyncMock(return_value={"thread_id": "thread-1"}),
        update_conversation_run_status=AsyncMock(),
        list_conversation_runs=AsyncMock(return_value=[run]),
    )
    return SimpleNamespace(store=store, graph=SimpleNamespace(aget_state=AsyncMock(return_value=saved)))


def mock_execution(monkeypatch, module):
    start = AsyncMock(return_value=RunStatus.COMPLETED)
    resume = AsyncMock(return_value=RunStatus.COMPLETED)

    async def persist(**kwargs):
        return kwargs["final_status"]

    finish = AsyncMock(side_effect=persist)
    if hasattr(module, "execute_message"):
        monkeypatch.setattr(module, "execute_message", start)
    monkeypatch.setattr(module, "execute_checkpoint_resume", resume)
    monkeypatch.setattr(module, "persist_run_outcome", finish)
    return start, resume, finish


@pytest.mark.parametrize("status", ["completed", "failed", "waiting_for_approval"])
def test_worker_finished_checkpoint_only_finalizes(monkeypatch, status):
    runtime = runtime_for(snapshot(status=status))
    start, resume, finish = mock_execution(monkeypatch, worker)
    assert asyncio.run(worker._execute(runtime, "request-1")) == status
    start.assert_not_awaited()
    resume.assert_not_awaited()
    assert finish.await_args.kwargs["final_status"] == RunStatus(status)
    runtime.store.finish_conversation_task_execution.assert_awaited_once_with("request-1", lease_token="execution-token", status="completed")


@pytest.mark.parametrize("attempts", [1, 2])
def test_owned_pending_checkpoint_resumes_regardless_of_claim_count(monkeypatch, attempts):
    runtime = runtime_for(snapshot(status="drafting", next_nodes=("model",)), attempts=attempts)
    start, resume, finish = mock_execution(monkeypatch, worker)
    asyncio.run(worker._execute(runtime, "request-1"))
    resume.assert_awaited_once()
    start.assert_not_awaited()
    finish.assert_not_awaited()


@pytest.mark.parametrize("saved", [
    snapshot(request_id="previous", status="drafting", next_nodes=("model",)),
    snapshot(session_id="other-session", next_nodes=("model",)),
    snapshot(request_id="previous"),
    SimpleNamespace(values={}, next=()),
])
def test_running_run_never_starts_or_resumes_without_owned_checkpoint(monkeypatch, saved):
    start, resume, finish = mock_execution(monkeypatch, worker)
    with pytest.raises(ValueError):
        asyncio.run(worker._execute(runtime_for(saved), "request-1"))
    start.assert_not_awaited()
    resume.assert_not_awaited()
    finish.assert_not_awaited()


@pytest.mark.parametrize("saved", [snapshot(request_id="previous"), SimpleNamespace(values={}, next=())])
def test_accepted_new_turn_can_start_with_previous_context_or_no_checkpoint(monkeypatch, saved):
    start, resume, finish = mock_execution(monkeypatch, worker)
    asyncio.run(worker._execute(runtime_for(saved, run_status="accepted"), "request-1"))
    start.assert_awaited_once()
    resume.assert_not_awaited()
    finish.assert_not_awaited()


@pytest.mark.parametrize("saved", [
    snapshot(request_id="previous", next_nodes=("model",)),
    snapshot(request_id="previous", session_id="other-session"),
    snapshot(status="drafting"),
])
def test_accepted_run_rejects_foreign_pending_or_invalid_finished_checkpoint(monkeypatch, saved):
    start, resume, finish = mock_execution(monkeypatch, worker)
    with pytest.raises(ValueError):
        asyncio.run(worker._execute(runtime_for(saved, run_status="accepted"), "request-1"))
    start.assert_not_awaited()
    resume.assert_not_awaited()
    finish.assert_not_awaited()


@pytest.mark.parametrize("status", ["completed", "failed", "waiting_for_approval"])
def test_startup_recovery_finalizes_all_terminal_states(monkeypatch, status):
    runtime = runtime_for(snapshot(status=status))
    _, resume, finish = mock_execution(monkeypatch, recovery)
    asyncio.run(recovery.recover_incomplete_runs(runtime))
    resume.assert_not_awaited()
    finish.assert_awaited_once()
    assert finish.await_args.kwargs["final_status"] == RunStatus(status)


def test_exhausted_retry_does_not_persist_another_requests_result(monkeypatch):
    runtime = runtime_for(snapshot(request_id="previous"))
    _, _, finish = mock_execution(monkeypatch, worker)
    asyncio.run(worker._prepare_retry(runtime, "request-1", ValueError("synthetic"), exhausted=True, lease_token="execution-token"))
    assert finish.await_args.kwargs["state"] is None
    assert finish.await_args.kwargs["final_status"] == RunStatus.FAILED


@pytest.mark.parametrize("saved", [snapshot(request_id="previous"), snapshot(session_id="other")])
def test_execution_error_fallback_does_not_read_foreign_state(saved):
    state = asyncio.run(execution._state_from_checkpoint(
        runtime_for(saved), "thread-1", request_id="request-1", session_id="session-1",
    ))
    assert state is None


@pytest.mark.parametrize("status", ["completed", "waiting_for_approval"])
def test_retry_after_result_commit_reuses_result_and_restores_privacy_only_once(monkeypatch, status):
    saved = snapshot(status=status)
    saved.values["draft"] = {"content": "Plan for <PERSON_NAME_1>"}
    saved.values["privacy_mapping_id"] = "synthetic-mapping-id"
    runtime = runtime_for(saved)
    results = {}

    async def get_result(request_id):
        return results.get(request_id)

    async def save_result(**kwargs):
        results[kwargs["request_id"]] = kwargs

    runtime.store.get_conversation_run_result = AsyncMock(side_effect=get_result)
    runtime.store.save_conversation_run_result = AsyncMock(side_effect=save_result)
    # Simulate a crash after the result commit but before the Run status commit.
    runtime.store.update_conversation_run_status = AsyncMock(side_effect=[RuntimeError("synthetic crash"), None])
    runtime.privacy_gateway_mode = "enforce"
    runtime.privacy_gateway_client = SimpleNamespace(
        restore=AsyncMock(return_value=SimpleNamespace(restored_text="Plan for Fictional Mia")),
    )
    start = AsyncMock()
    resume = AsyncMock()
    monkeypatch.setattr(worker, "execute_message", start)
    monkeypatch.setattr(worker, "execute_checkpoint_resume", resume)
    monkeypatch.setattr(execution, "sync_workspace_checkpoint", AsyncMock())
    publish = AsyncMock()
    monkeypatch.setattr(execution, "_publish_state_events", publish)

    with pytest.raises(RuntimeError, match="synthetic crash"):
        asyncio.run(worker._execute(runtime, "request-1"))
    assert asyncio.run(worker._execute(runtime, "request-1")) == status
    start.assert_not_awaited()
    resume.assert_not_awaited()
    runtime.privacy_gateway_client.restore.assert_awaited_once()
    runtime.store.save_conversation_run_result.assert_awaited_once()
    publish.assert_awaited_once()
    assert results["request-1"]["draft"]["content"] == "Plan for Fictional Mia"
    assert saved.values["draft"]["content"] == "Plan for <PERSON_NAME_1>"
