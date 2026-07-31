import sqlite3

from fastapi.testclient import TestClient

from app.api import ApiRuntime, build_api_runtime
from app.main import create_app


def test_api_runtime_builds_shared_store_checkpointer_and_graph(tmp_path) -> None:
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )

    try:
        assert runtime.store.get_class_profile("kangaroo-room") is not None
        assert runtime.graph.checkpointer is runtime.checkpointer
        assert runtime.is_closed is False
        assert (tmp_path / "eduflow.sqlite3").exists()
        assert (tmp_path / "checkpoints.sqlite3").exists()
    finally:
        runtime.close()


def test_api_runtime_close_is_idempotent_and_closes_checkpoint_connection(
    tmp_path,
) -> None:
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )

    runtime.close()
    runtime.close()

    assert runtime.is_closed is True
    try:
        runtime.checkpointer.conn.execute("SELECT 1")
    except sqlite3.ProgrammingError as error:
        assert "closed" in str(error).lower()
    else:
        raise AssertionError("Checkpoint connection should be closed")


def test_fastapi_lifespan_attaches_and_closes_runtime(tmp_path) -> None:
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    application = create_app(runtime_factory=lambda: runtime)

    with TestClient(application) as client:
        assert application.state.runtime is runtime
        assert runtime.is_closed is False
        assert client.get("/health").status_code == 200

    assert runtime.is_closed is True
