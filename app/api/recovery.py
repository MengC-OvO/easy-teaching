"""Reconcile and resume durable conversation runs when the API starts."""

from app.api.execution import execute_checkpoint_resume, persist_run_outcome
from app.api.runtime import ApiRuntime
from app.schemas import GraphState, RunStatus, WorkflowStatus
from app.workflows import checkpoint_config


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
    conversation = await runtime.store.get_conversation_session(session_id)
    if conversation is None:
        raise ValueError("Conversation session does not exist")

    thread_id = conversation["thread_id"]
    snapshot = await runtime.graph.aget_state(checkpoint_config(thread_id))
    if not snapshot.values:
        raise ValueError("Recovery checkpoint does not exist")

    state = GraphState.model_validate(snapshot.values)
    if state.request_id != request_id or state.session_id != session_id:
        raise ValueError("Recovery checkpoint belongs to another run")

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

    final_status = _completed_snapshot_status(state)
    await persist_run_outcome(
        runtime=runtime,
        request_id=request_id,
        session_id=session_id,
        state=state,
        final_status=final_status,
    )


def _completed_snapshot_status(state: GraphState) -> RunStatus:
    if state.workflow_status is WorkflowStatus.COMPLETED:
        return RunStatus.COMPLETED
    if state.workflow_status is WorkflowStatus.FAILED:
        return RunStatus.FAILED
    raise ValueError("Checkpoint has no next node but the run is not terminal")
