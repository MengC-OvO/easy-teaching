from fastapi.testclient import TestClient

from app.api import build_api_runtime
from app.main import create_app
from app.schemas import GraphState, WorkflowStatus


class RecordingGraph:
    def __init__(self, workflow_status=WorkflowStatus.COMPLETED) -> None:
        self.workflow_status = workflow_status
        self.calls = []

    def invoke(self, state, config):
        self.calls.append((state, config))
        return GraphState.model_validate(state).model_copy(
            update={"workflow_status": self.workflow_status}
        )


def _client(tmp_path, graph=None):
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    if graph is not None:
        runtime.graph = graph
    application = create_app(runtime_factory=lambda: runtime)
    return runtime, TestClient(application)


def test_message_is_accepted_and_executes_on_the_session_thread(tmp_path) -> None:
    graph = RecordingGraph()
    runtime, client = _client(tmp_path, graph)

    with client:
        session = client.post(
            "/sessions",
            json={"teacher_id": "teacher-001", "class_id": "kangaroo-room"},
        ).json()
        response = client.post(
            f"/sessions/{session['session_id']}/messages",
            json={"message": "Plan an outdoor activity."},
        )

        assert response.status_code == 202
        body = response.json()
        assert body["session_id"] == session["session_id"]
        assert body["status"] == "accepted"

        state, config = graph.calls[0]
        assert state.request_id == body["request_id"]
        assert state.session_id == session["session_id"]
        assert state.thread_id == session["thread_id"]
        assert state.teacher_id == "teacher-001"
        assert state.class_id == "kangaroo-room"
        assert state.user_message == "Plan an outdoor activity."
        assert config == {"configurable": {"thread_id": session["thread_id"]}}
        assert runtime.store.get_conversation_run(body["request_id"])["status"] == "completed"


def test_client_request_id_is_idempotent(tmp_path) -> None:
    graph = RecordingGraph()
    _, client = _client(tmp_path, graph)

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        payload = {"message": "Plan an activity.", "request_id": "req-client-001"}

        first = client.post(f"/sessions/{session_id}/messages", json=payload)
        second = client.post(f"/sessions/{session_id}/messages", json=payload)

        assert first.status_code == 202
        assert first.json()["status"] == "accepted"
        assert second.status_code == 202
        assert second.json()["status"] == "completed"
        assert len(graph.calls) == 1


def test_message_for_missing_session_returns_404(tmp_path) -> None:
    _, client = _client(tmp_path, RecordingGraph())

    with client:
        response = client.post(
            "/sessions/missing/messages",
            json={"message": "Plan an activity.", "request_id": "req-missing"},
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "session_not_found"
        assert response.json()["request_id"] == "req-missing"


def test_waiting_run_blocks_a_new_message_on_the_same_thread(tmp_path) -> None:
    graph = RecordingGraph(WorkflowStatus.WAITING_FOR_APPROVAL)
    _, client = _client(tmp_path, graph)

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        first = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Write a learning record.", "request_id": "req-waiting"},
        )
        second = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Start something else.", "request_id": "req-new"},
        )

        assert first.status_code == 202
        assert second.status_code == 409
        assert second.json()["error"]["code"] == "session_busy"
        assert second.json()["error"]["details"] == {
            "active_request_id": "req-waiting"
        }

