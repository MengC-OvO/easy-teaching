from uuid import UUID

from fastapi.testclient import TestClient

from app.api import build_api_runtime
from app.main import create_app


def _runtime(tmp_path):
    return build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )


def test_create_session_persists_scope_and_returns_stable_ids(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    application = create_app(runtime_factory=lambda: runtime)

    with TestClient(application) as client:
        response = client.post(
            "/sessions",
            json={"teacher_id": "teacher-001", "class_id": "kangaroo-room"},
        )

        assert response.status_code == 201
        body = response.json()
        UUID(body["session_id"])
        UUID(body["thread_id"])
        assert body["session_id"] != body["thread_id"]
        assert body["status"] == "active"

        stored = runtime.store.get_conversation_session(body["session_id"])
        assert stored is not None
        assert stored["teacher_id"] == "teacher-001"
        assert stored["class_id"] == "kangaroo-room"
        assert stored["thread_id"] == body["thread_id"]


def test_get_session_returns_the_ids_created_by_post(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    application = create_app(runtime_factory=lambda: runtime)

    with TestClient(application) as client:
        created = client.post("/sessions", json={}).json()
        response = client.get(f"/sessions/{created['session_id']}")

        assert response.status_code == 200
        assert response.json() == created


def test_get_missing_session_returns_structured_error(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    application = create_app(runtime_factory=lambda: runtime)

    with TestClient(application) as client:
        response = client.get("/sessions/missing-session")

        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "session_not_found",
                "message": "The requested session does not exist.",
                "recoverable": False,
                "details": {"session_id": "missing-session"},
            },
            "request_id": None,
        }


def test_session_is_available_after_store_is_reopened(tmp_path) -> None:
    database_path = tmp_path / "eduflow.sqlite3"
    first_runtime = build_api_runtime(
        database_path=database_path,
        checkpoint_database_path=tmp_path / "checkpoints-one.sqlite3",
    )
    application = create_app(runtime_factory=lambda: first_runtime)

    with TestClient(application) as client:
        created = client.post("/sessions", json={"teacher_id": "teacher-001"}).json()

    second_runtime = build_api_runtime(
        database_path=database_path,
        checkpoint_database_path=tmp_path / "checkpoints-two.sqlite3",
    )
    try:
        stored = second_runtime.store.get_conversation_session(created["session_id"])
        assert stored is not None
        assert stored["thread_id"] == created["thread_id"]
        assert stored["teacher_id"] == "teacher-001"
    finally:
        second_runtime.close()

