from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient

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
from app.services import ConversationSessionBusyError, EduFlowStore


class WaitingGraph:
    def __init__(self) -> None:
        self.initial_calls = []
        self.resume_calls = []

    def invoke(self, value, config):
        if not isinstance(value, GraphState):
            self.resume_calls.append((value, config))
            return self.paused_state.model_copy(
                update={"workflow_status": WorkflowStatus.COMPLETED}
            )
        self.initial_calls.append((value, config))
        self.paused_state = value.model_copy(
            update={
                "workflow_status": WorkflowStatus.WAITING_FOR_APPROVAL,
                "draft": Draft(title="Draft", content="Content"),
                "approval": Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                ),
            }
        )
        return self.paused_state


def test_database_allows_only_one_active_run_per_session(tmp_path) -> None:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'concurrency.sqlite3'}")
    store.initialize()
    store.create_conversation_session(
        session_id="session-concurrent",
        thread_id="thread-concurrent",
        teacher_id=None,
        class_id=None,
    )
    barrier = Barrier(2)

    def create_run(request_id):
        barrier.wait()
        try:
            result = store.create_conversation_run(
                request_id=request_id,
                session_id="session-concurrent",
            )
            return ("created", result["request_id"])
        except ConversationSessionBusyError as error:
            return ("busy", error.active_request_id)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(create_run, ["req-concurrent-a", "req-concurrent-b"])
            )

        assert sorted(result[0] for result in results) == ["busy", "created"]
        active = store.get_active_conversation_run("session-concurrent")
        assert active["request_id"] in {"req-concurrent-a", "req-concurrent-b"}
    finally:
        store.engine.dispose()


def test_session_accepts_a_new_run_after_the_previous_run_finishes(tmp_path) -> None:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'sequential.sqlite3'}")
    store.initialize()
    store.create_conversation_session(
        session_id="session-sequential",
        thread_id="thread-sequential",
        teacher_id=None,
        class_id=None,
    )

    try:
        store.create_conversation_run(
            request_id="req-first",
            session_id="session-sequential",
        )
        store.update_conversation_run_status("req-first", "completed")

        second = store.create_conversation_run(
            request_id="req-second",
            session_id="session-sequential",
        )

        assert second["created"] is True
        assert second["request_id"] == "req-second"
    finally:
        store.engine.dispose()


def test_message_race_does_not_schedule_the_same_request_twice(
    tmp_path,
    monkeypatch,
) -> None:
    graph = WaitingGraph()
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    runtime.graph = graph
    application = create_app(runtime_factory=lambda: runtime)

    with TestClient(application) as client:
        session = client.post("/sessions", json={}).json()
        runtime.store.create_conversation_run(
            request_id="req-message-race",
            session_id=session["session_id"],
        )
        monkeypatch.setattr(runtime.store, "get_conversation_run", lambda _: None)
        monkeypatch.setattr(runtime.store, "get_active_conversation_run", lambda _: None)

        response = client.post(
            f"/sessions/{session['session_id']}/messages",
            json={"message": "Retry message.", "request_id": "req-message-race"},
        )

        assert response.status_code == 202
        assert response.json()["status"] == "accepted"
        assert graph.initial_calls == []
        assert runtime.store.list_conversation_events(
            request_id="req-message-race"
        ) == []


def test_approval_race_does_not_resume_the_graph_twice(tmp_path, monkeypatch) -> None:
    graph = WaitingGraph()
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    runtime.graph = graph
    application = create_app(runtime_factory=lambda: runtime)

    with TestClient(application) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Create a draft.", "request_id": "req-approval-race"},
        )
        runtime.store.create_approval_decision(
            request_id="req-approval-race",
            session_id=session_id,
            decision="approve",
        )
        monkeypatch.setattr(runtime.store, "get_approval_decision", lambda _: None)

        response = client.post(
            f"/sessions/{session_id}/approvals",
            json={"request_id": "req-approval-race", "decision": "approve"},
        )

        assert response.status_code == 202
        assert response.json()["status"] == "waiting_for_approval"
        assert graph.resume_calls == []
