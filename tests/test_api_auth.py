import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.api.auth as auth_module
from app.api import build_api_runtime
from app.api.auth import CurrentUser, SupabaseAuthClient, get_current_user
from app.main import create_app


def _runtime(tmp_path):
    return build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )


def test_supabase_auth_client_returns_trusted_user() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == "publishable-key"
        assert request.headers["authorization"] == "Bearer valid-token"
        return httpx.Response(
            200,
            json={"id": "auth-user-001", "email": "teacher@example.com"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = SupabaseAuthClient(
            base_url="https://project.supabase.co",
            publishable_key="publishable-key",
            http_client=http_client,
        )
        user = client.get_user("valid-token")

    assert user.teacher_id == "auth-user-001"
    assert user.email == "teacher@example.com"


def test_supabase_auth_client_rejects_invalid_token() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(401, json={}))
    with httpx.Client(transport=transport) as http_client:
        client = SupabaseAuthClient(
            base_url="https://project.supabase.co",
            publishable_key="publishable-key",
            http_client=http_client,
        )
        with pytest.raises(HTTPException) as caught:
            client.get_user("invalid-token")

    assert caught.value.status_code == 401


def test_authenticated_session_uses_current_user_and_enforces_owner(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    application = create_app(runtime_factory=lambda: runtime)
    application.dependency_overrides[get_current_user] = lambda: CurrentUser(
        user_id="auth-user-001",
        email="teacher@example.com",
    )

    with TestClient(application) as client:
        created = client.post(
            "/sessions",
            json={"teacher_id": "forged-user", "class_id": "kangaroo-room"},
        )
        assert created.status_code == 201
        stored = runtime.store.get_conversation_session(created.json()["session_id"])
        assert stored["teacher_id"] == "auth-user-001"

        application.dependency_overrides[get_current_user] = lambda: CurrentUser(
            user_id="another-user",
            email="other@example.com",
        )
        forbidden = client.get(f"/sessions/{created.json()['session_id']}")

    assert forbidden.status_code == 403
    assert forbidden.json()["detail"] == "You do not have access to this session."


def test_auth_session_sets_http_only_cookie_and_returns_current_user(
    tmp_path,
    monkeypatch,
) -> None:
    runtime = _runtime(tmp_path)
    application = create_app(runtime_factory=lambda: runtime)

    class FakeAuthClient:
        def get_user(self, access_token: str) -> CurrentUser:
            assert access_token == "valid-token"
            return CurrentUser("auth-user-001", "teacher@example.com")

    monkeypatch.setattr(auth_module.settings, "auth_enabled", True)
    monkeypatch.setattr(auth_module, "_auth_client", lambda: FakeAuthClient())

    with TestClient(application) as client:
        response = client.post("/auth/session", json={"access_token": "valid-token"})
        assert response.status_code == 200
        assert response.json() == {
            "user_id": "auth-user-001",
            "email": "teacher@example.com",
        }
        assert "HttpOnly" in response.headers["set-cookie"]

        current = client.get("/auth/me")
        assert current.status_code == 200
        assert current.json()["user_id"] == "auth-user-001"

        logged_out = client.delete("/auth/session")
        assert logged_out.status_code == 204
        assert auth_module.AUTH_COOKIE_NAME not in client.cookies
