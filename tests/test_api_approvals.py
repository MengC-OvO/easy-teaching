from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api import build_api_runtime
from app.main import create_app
from app.schemas import (
    Approval,
    ApprovalStatus,
    Draft,
    GraphState,
    RiskLevel,
    WorkflowStatus,
)


class InterruptThenResumeGraph:
    def __init__(self) -> None:
        self.paused_state = None
        self.resume_calls = []

    def invoke(self, value, config):
        if isinstance(value, Command):
            self.resume_calls.append((value, config))
            decision = value.resume["decision"]
            approval_status = (
                ApprovalStatus.APPROVED
                if decision == "approve"
                else ApprovalStatus.REJECTED
            )
            return self.paused_state.model_copy(
                update={
                    "workflow_status": WorkflowStatus.COMPLETED,
                    "approval": self.paused_state.approval.model_copy(
                        update={"status": approval_status}
                    ),
                }
            )

        self.paused_state = GraphState.model_validate(value).model_copy(
            update={
                "workflow_status": WorkflowStatus.WAITING_FOR_APPROVAL,
                "draft": Draft(title="Learning record", content="Draft content"),
                "approval": Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason="Teacher review is required.",
                ),
            }
        )
        return self.paused_state


def _client(tmp_path):
    graph = InterruptThenResumeGraph()
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    runtime.graph = graph
    application = create_app(runtime_factory=lambda: runtime)
    return graph, TestClient(application)


def _waiting_request(client):
    session = client.post("/sessions", json={}).json()
    client.post(
        f"/sessions/{session['session_id']}/messages",
        json={"message": "Write a learning record.", "request_id": "req-approval"},
    )
    return session


def test_approval_resumes_the_same_thread_and_refreshes_draft_snapshot(tmp_path) -> None:
    graph, client = _client(tmp_path)

    with client:
        session = _waiting_request(client)
        response = client.post(
            f"/sessions/{session['session_id']}/approvals",
            json={"request_id": "req-approval", "decision": "approve"},
        )

        assert response.status_code == 202
        assert response.json() == {
            "session_id": session["session_id"],
            "request_id": "req-approval",
            "decision": "approve",
            "status": "running",
        }
        command, config = graph.resume_calls[0]
        assert command.resume == {
            "request_id": "req-approval",
            "decision": "approve",
        }
        assert config == {"configurable": {"thread_id": session["thread_id"]}}

        draft = client.get(
            f"/sessions/{session['session_id']}/drafts/req-approval"
        ).json()
        assert draft["status"] == "completed"
        assert draft["approval"]["status"] == "approved"


def test_same_approval_retry_is_idempotent_but_different_decision_conflicts(tmp_path) -> None:
    graph, client = _client(tmp_path)

    with client:
        session = _waiting_request(client)
        url = f"/sessions/{session['session_id']}/approvals"

        first = client.post(
            url,
            json={"request_id": "req-approval", "decision": "reject"},
        )
        retry = client.post(
            url,
            json={"request_id": "req-approval", "decision": "reject"},
        )
        conflict = client.post(
            url,
            json={"request_id": "req-approval", "decision": "approve"},
        )

        assert first.status_code == 202
        assert retry.status_code == 202
        assert retry.json()["status"] == "completed"
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "approval_decision_conflict"
        assert len(graph.resume_calls) == 1


def test_approval_requires_a_waiting_request_in_the_same_session(tmp_path) -> None:
    _, client = _client(tmp_path)

    with client:
        first_session = client.post("/sessions", json={}).json()["session_id"]
        second_session = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{first_session}/messages",
            json={"message": "Write a record.", "request_id": "req-approval"},
        )

        response = client.post(
            f"/sessions/{second_session}/approvals",
            json={"request_id": "req-approval", "decision": "approve"},
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "request_not_found"

