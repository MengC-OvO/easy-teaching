from types import SimpleNamespace

from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api import build_api_runtime
from app.api.recovery import recover_incomplete_runs
from app.main import create_app
from app.schemas import (
    Approval,
    ApprovalStatus,
    Draft,
    GraphState,
    RiskLevel,
    RunStatus,
    WorkflowStatus,
)


class RecoveryGraph:
    def __init__(self, checkpoint_state, *, next_nodes=()) -> None:
        self.checkpoint_state = checkpoint_state
        self.next_nodes = next_nodes
        self.invoke_calls = []

    def get_state(self, config):
        values = (
            {}
            if self.checkpoint_state is None
            else self.checkpoint_state.model_dump(mode="json")
        )
        return SimpleNamespace(values=values, next=self.next_nodes)

    def invoke(self, value, config):
        self.invoke_calls.append((value, config))
        approval = self.checkpoint_state.approval
        if isinstance(value, Command):
            approval = approval.model_copy(
                update={"status": ApprovalStatus.APPROVED}
            )
        self.checkpoint_state = self.checkpoint_state.model_copy(
            update={
                "workflow_status": WorkflowStatus.COMPLETED,
                "approval": approval,
            }
        )
        self.next_nodes = ()
        return self.checkpoint_state


def _runtime_with_run(tmp_path, *, request_id, run_status, state, next_nodes=()):
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    session_id = "session-recovery"
    thread_id = "thread-recovery"
    runtime.store.create_conversation_session(
        session_id=session_id,
        thread_id=thread_id,
        teacher_id=None,
        class_id=None,
    )
    runtime.store.create_conversation_run(
        request_id=request_id,
        session_id=session_id,
    )
    runtime.store.update_conversation_run_status(request_id, run_status.value)
    runtime.graph = RecoveryGraph(state, next_nodes=next_nodes)
    return runtime, session_id, thread_id


def _state(*, request_id, workflow_status, with_draft=False):
    return GraphState(
        request_id=request_id,
        session_id="session-recovery",
        thread_id="thread-recovery",
        user_message="Create a safe activity plan.",
        workflow_status=workflow_status,
        draft=(
            Draft(title="Recovered plan", content="Recovered draft content")
            if with_draft
            else None
        ),
        approval=(
            Approval(
                status=ApprovalStatus.REQUIRED,
                risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                reason="Teacher review is required.",
            )
            if with_draft
            else Approval()
        ),
    )


def test_recovery_keeps_an_unapproved_run_waiting(tmp_path) -> None:
    request_id = "req-waiting-recovery"
    runtime, session_id, _ = _runtime_with_run(
        tmp_path,
        request_id=request_id,
        run_status=RunStatus.WAITING_FOR_APPROVAL,
        state=_state(
            request_id=request_id,
            workflow_status=WorkflowStatus.WAITING_FOR_APPROVAL,
            with_draft=True,
        ),
        next_nodes=("approval_gate",),
    )

    try:
        recover_incomplete_runs(runtime)

        assert runtime.graph.invoke_calls == []
        assert runtime.store.get_conversation_run(request_id)["status"] == (
            RunStatus.WAITING_FOR_APPROVAL.value
        )
        result = runtime.store.get_conversation_run_result(request_id)
        assert result["session_id"] == session_id
        assert result["draft"]["title"] == "Recovered plan"
    finally:
        runtime.close()


def test_recovery_continues_a_checkpoint_with_a_next_node(tmp_path) -> None:
    request_id = "req-next-node-recovery"
    runtime, _, thread_id = _runtime_with_run(
        tmp_path,
        request_id=request_id,
        run_status=RunStatus.RUNNING,
        state=_state(
            request_id=request_id,
            workflow_status=WorkflowStatus.ROUTED,
        ),
        next_nodes=("planning",),
    )

    try:
        recover_incomplete_runs(runtime)

        resumed_value, config = runtime.graph.invoke_calls[0]
        assert resumed_value is None
        assert config == {"configurable": {"thread_id": thread_id}}
        assert runtime.store.get_conversation_run(request_id)["status"] == (
            RunStatus.COMPLETED.value
        )
    finally:
        runtime.close()


def test_recovery_resumes_a_saved_approval_decision(tmp_path) -> None:
    request_id = "req-approval-recovery"
    runtime, session_id, _ = _runtime_with_run(
        tmp_path,
        request_id=request_id,
        run_status=RunStatus.WAITING_FOR_APPROVAL,
        state=_state(
            request_id=request_id,
            workflow_status=WorkflowStatus.WAITING_FOR_APPROVAL,
            with_draft=True,
        ),
        next_nodes=("approval_gate",),
    )
    runtime.store.create_approval_decision(
        request_id=request_id,
        session_id=session_id,
        decision="approve",
    )

    try:
        recover_incomplete_runs(runtime)

        resumed_value, _ = runtime.graph.invoke_calls[0]
        assert isinstance(resumed_value, Command)
        assert runtime.store.get_conversation_run(request_id)["status"] == (
            RunStatus.COMPLETED.value
        )
    finally:
        runtime.close()


def test_recovery_reconciles_a_terminal_checkpoint_without_rerunning(tmp_path) -> None:
    request_id = "req-completed-recovery"
    runtime, _, _ = _runtime_with_run(
        tmp_path,
        request_id=request_id,
        run_status=RunStatus.RUNNING,
        state=_state(
            request_id=request_id,
            workflow_status=WorkflowStatus.COMPLETED,
            with_draft=True,
        ),
    )

    try:
        recover_incomplete_runs(runtime)

        assert runtime.graph.invoke_calls == []
        assert runtime.store.get_conversation_run(request_id)["status"] == (
            RunStatus.COMPLETED.value
        )
    finally:
        runtime.close()


def test_recovery_marks_a_run_failed_when_its_checkpoint_is_missing(tmp_path) -> None:
    request_id = "req-missing-checkpoint"
    runtime, _, _ = _runtime_with_run(
        tmp_path,
        request_id=request_id,
        run_status=RunStatus.ACCEPTED,
        state=None,
    )

    try:
        recover_incomplete_runs(runtime)

        assert runtime.graph.invoke_calls == []
        assert runtime.store.get_conversation_run(request_id)["status"] == (
            RunStatus.FAILED.value
        )
        events = runtime.store.list_conversation_events(request_id=request_id)
        assert events[-1]["event"] == "failed"
    finally:
        runtime.close()


def test_fastapi_startup_runs_recovery_before_serving_requests(tmp_path) -> None:
    request_id = "req-startup-recovery"
    runtime, _, _ = _runtime_with_run(
        tmp_path,
        request_id=request_id,
        run_status=RunStatus.RUNNING,
        state=_state(
            request_id=request_id,
            workflow_status=WorkflowStatus.ROUTED,
        ),
        next_nodes=("planning",),
    )
    application = create_app(runtime_factory=lambda: runtime)

    with TestClient(application) as client:
        assert client.get("/health").status_code == 200
        assert runtime.store.get_conversation_run(request_id)["status"] == (
            RunStatus.COMPLETED.value
        )
        assert len(runtime.graph.invoke_calls) == 1

    assert runtime.is_closed is True
