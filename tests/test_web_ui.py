from fastapi.testclient import TestClient

from app.api import build_api_runtime
from app.main import create_app


def _test_app(tmp_path):
    return create_app(
        runtime_factory=lambda: build_api_runtime(
            database_path=tmp_path / "eduflow.sqlite3",
            checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
        )
    )


def test_root_serves_teacher_workspace(tmp_path) -> None:
    with TestClient(_test_app(tmp_path)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "EduFlow AU" in response.text
    assert "Message EduFlow" in response.text


def test_web_assets_are_served(tmp_path) -> None:
    with TestClient(_test_app(tmp_path)) as client:
        stylesheet = client.get("/assets/styles.css")
        javascript = client.get("/assets/app.js")

    assert stylesheet.status_code == 200
    assert "--purple" in stylesheet.text
    assert javascript.status_code == 200
    assert "EventSource" in javascript.text
