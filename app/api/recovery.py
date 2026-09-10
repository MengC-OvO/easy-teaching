"""Reconcile and resume durable conversation runs when the API starts."""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.api.checkpoint_config import checkpoint_config
from app.api.execution import execute_checkpoint_resume, persist_run_outcome
from app.api.recovery_state import checkpoint_belongs_to_run, completed_snapshot_status
from app.schemas import GraphState, RunStatus
if TYPE_CHECKING:
    from app.api.runtime import ApiRuntime


_INCOMPLETE_STATUSES = [
    RunStatus.ACCEPTED.value,
    RunStatus.RUNNING.value,
]


async def recover_incomplete_runs(runtime: ApiRuntime) -> None:
    """Recover every non-terminal run without preventing application startup."""
    for run in await runtime.store.list_conversation_runs(statuses=_INCOMPLETE_STATUSES):
        try:
            await _recover_run(runtime, run)
        except Exception:
            await persist_run_outcome(
                runtime=runtime,
                request_id=run["request_id"],
                session_id=run["session_id"],
                state=None,
                final_status=RunStatus.FAILED,
            )


async def _recover_run(runtime: ApiRuntime, run: dict) -> None:
    request_id = run["request_id"]
    session_id = run["session_id"]
    task_reader = getattr(runtime.store, "get_conversation_task", None)
    if task_reader is not None:
        task = await task_reader(request_id)
        if task and task["payload"].get("kind") == "approved_action":
            # The completed graph checkpoint describes the pre-approval phase.
            # Its outcome must not overwrite an admitted action's durable state.
            return
    conversation = await runtime.store.get_conversation_session(session_id)
    if conversation is None:
        raise ValueError("Conversation session does not exist")

    thread_id = conversation["thread_id"]
    snapshot = await runtime.graph.aget_state(checkpoint_config(thread_id))
    if not snapshot.values:
        raise ValueError("Recovery checkpoint does not exist")

    if not checkpoint_belongs_to_run(
        snapshot.values, request_id=request_id, session_id=session_id
    ):
        raise ValueError("Recovery checkpoint belongs to another run")
    state = GraphState.model_validate(snapshot.values)

    if snapshot.next:
        await runtime.store.update_conversation_run_status(
            request_id,
            RunStatus.RUNNING.value,
        )
        await execute_checkpoint_resume(
            runtime=runtime,
            request_id=request_id,
            session_id=session_id,
            thread_id=thread_id,
        )
        return

    final_status = completed_snapshot_status(state)
    await persist_run_outcome(
        runtime=runtime,
        request_id=request_id,
        session_id=session_id,
        state=state,
        final_status=final_status,
    )


_completed_snapshot_status = completed_snapshot_status
