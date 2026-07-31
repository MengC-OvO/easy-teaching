from fastapi.testclient import TestClient

from app.api import build_api_runtime
from app.main import create_app
from app.schemas import (
    Approval,
    ApprovalStatus,
    Citation,
    Draft,
    GraphState,
    RiskLevel,
    WorkflowStatus,
)


class DraftGraph:
    def invoke(self, state, config):
        return GraphState.model_validate(state).model_copy(
            update={
                "workflow_status": WorkflowStatus.WAITING_FOR_APPROVAL,
                "draft": Draft(title="Outdoor plan", content="Draft content"),
                "approval": Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason="Teacher review is required.",
                ),
                "citations": [
                    Citation(source="eylf-v2", section="Outcome 4"),
                ],
            }
        )


class NoDraftGraph:
    def invoke(self, state, config):
        return GraphState.model_validate(state).model_copy(
            update={"workflow_status": WorkflowStatus.COMPLETED}
        )


def _client(tmp_path, graph):
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    runtime.graph = graph
    application = create_app(runtime_factory=lambda: runtime)
    return TestClient(application)


def test_get_draft_returns_request_snapshot_with_approval_and_citations(tmp_path) -> None:
    client = _client(tmp_path, DraftGraph())

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        accepted = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Write an outdoor plan.", "request_id": "req-draft"},
        ).json()
        response = client.get(
            f"/sessions/{session_id}/drafts/{accepted['request_id']}"
        )

        assert response.status_code == 200
        assert response.json() == {
            "session_id": session_id,
            "request_id": "req-draft",
            "status": "waiting_for_approval",
            "draft": {
                "title": "Outdoor plan",
                "content": "Draft content",
                "is_draft": True,
            },
            "approval": {
                "status": "required",
                "risk_level": "L2_controlled_write",
                "reason": "Teacher review is required.",
            },
            "citations": [
                {
                    "source": "eylf-v2",
                    "title": None,
                    "section": "Outcome 4",
                    "page": None,
                    "url": None,
                }
            ],
        }


def test_draft_lookup_does_not_cross_session_boundary(tmp_path) -> None:
    client = _client(tmp_path, DraftGraph())

    with client:
        first_session = client.post("/sessions", json={}).json()["session_id"]
        second_session = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{first_session}/messages",
            json={"message": "Write a plan.", "request_id": "req-private"},
        )
        response = client.get(
            f"/sessions/{second_session}/drafts/req-private"
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "request_not_found"


def test_completed_run_without_draft_returns_not_found(tmp_path) -> None:
    client = _client(tmp_path, NoDraftGraph())

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Answer a question.", "request_id": "req-no-draft"},
        )
        response = client.get(
            f"/sessions/{session_id}/drafts/req-no-draft"
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "draft_not_found"

