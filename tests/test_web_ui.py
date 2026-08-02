from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_teacher_workspace() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "EduFlow AU" in response.text
    assert "Message EduFlow" in response.text


def test_web_assets_are_served() -> None:
    with TestClient(app) as client:
        stylesheet = client.get("/assets/styles.css")
        javascript = client.get("/assets/app.js")

    assert stylesheet.status_code == 200
    assert "--purple" in stylesheet.text
    assert javascript.status_code == 200
    assert "EventSource" in javascript.text
