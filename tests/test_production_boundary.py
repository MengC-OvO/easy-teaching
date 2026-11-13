import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.auth import CurrentUser, SupabaseAuthClient, get_current_user, _request_token
from app.api.security import RequestBoundaryMiddleware
from app.config import Settings, settings
from app.main import create_app
from app.services.async_store import AsyncEasyTeachingStore
from app.services.models import Base, CentreRecord, ClassRecord, TeacherRecord, TeacherClassMembershipRecord
from tests.test_health import ReadyRuntime
from tests.test_outbox_leases import AsyncSessionShim


def production_settings(**overrides):
    values = dict(APP_ENV="production", AUTH_ENABLED=True, PUBLIC_ORIGIN="https://teaching.example.org",
                  SUPABASE_URL="https://auth.example.org", SUPABASE_PUBLISHABLE_KEY="test-public-key",
                  DATABASE_URL="postgresql+psycopg://user:password@db/app",
                  CHECKPOINT_DATABASE_URL="postgresql://user:password@db/app",
                  MODEL_BASE_URL="https://model.example.org/v1", MODEL_API_KEY="test-model-key",
                  EMBEDDING_API_KEY="test-embedding-key")
    return Settings(_env_file=None, **(values | overrides))


@pytest.mark.parametrize("overrides", [
    {"AUTH_ENABLED": False}, {"PUBLIC_ORIGIN": "http://example.org"},
    {"PUBLIC_ORIGIN": "https://example.org/path"}, {"PUBLIC_ORIGIN": "https://user:secret@example.org"},
    {"MODEL_API_KEY": "replace-with-your-local-key"}, {"SUPABASE_URL": "http://example.org"},
    {"TASK_EXECUTION_MODE": "inline"}, {"API_RATE_LIMIT_ENABLED": False},
])
def test_production_rejects_incomplete_configuration(overrides):
    with pytest.raises(ValueError):
        production_settings(**overrides)


def test_complete_production_configuration():
    assert production_settings().auth_enabled


def boundary_app():
    app = FastAPI()
    app.add_middleware(RequestBoundaryMiddleware)

    @app.post("/echo")
    async def echo(request: Request):
        return {"size": len(await request.body())}

    return app


@pytest.mark.parametrize("headers", [{"Origin": "https://attacker.example"}, {"Sec-Fetch-Site": "cross-site"}, {"Origin": "null"}])
def test_cross_site_mutations_are_rejected(headers):
    assert TestClient(boundary_app()).post("/echo", headers=headers).status_code == 403


def test_same_origin_and_service_requests_work(monkeypatch):
    monkeypatch.setattr(settings, "public_origin", "https://school.example")
    client = TestClient(boundary_app())
    assert client.post("/echo", headers={"Origin": "https://school.example"}, content=b"ok").status_code == 200
    assert client.post("/echo", content=b"ok").status_code == 200


def test_body_limit_checks_declared_and_chunked_bytes(monkeypatch):
    monkeypatch.setattr(settings, "request_max_bytes", 10)
    client = TestClient(boundary_app())
    assert client.post("/echo", content=b"x" * 11).status_code == 413
    assert client.post("/echo", content=iter([b"123456", b"78901"])).status_code == 413
    assert client.post("/echo", content=b"1234567890").json() == {"size": 10}


def test_chunked_multipart_limit_aborts_parsing(monkeypatch):
    from fastapi import File, UploadFile
    app = boundary_app()

    @app.post("/upload")
    async def upload(file: UploadFile = File()):
        pytest.fail("Oversized multipart must not reach the upload handler")

    monkeypatch.setattr(settings, "request_max_bytes", 150)
    prefix = b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.txt"\r\n\r\n'
    response = TestClient(app).post("/upload", headers={"Content-Type": "multipart/form-data; boundary=boundary"},
                                    content=iter([prefix, b"x" * 200, b"\r\n--boundary--\r\n"]))
    assert response.status_code == 413


def test_private_responses_are_not_cached_or_framed():
    response = TestClient(create_app()).get("/health")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert "script-src 'self'" in response.headers["content-security-policy"]


def test_readiness_times_out_and_reports_failed_postgres(monkeypatch):
    runtime = ReadyRuntime()

    async def stuck():
        await asyncio.sleep(5)

    runtime.store.healthcheck = stuck
    monkeypatch.setattr(settings, "readiness_timeout_seconds", 0.01)
    with TestClient(create_app(lambda: runtime)) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["postgres"] == "unavailable"


def test_runtime_closes_when_recovery_fails(monkeypatch):
    import app.main as main
    runtime = ReadyRuntime()
    runtime.close = AsyncMock()
    monkeypatch.setattr(main, "recover_incomplete_runs", AsyncMock(side_effect=RuntimeError("startup failure")))
    with pytest.raises(RuntimeError, match="startup failure"):
        with TestClient(create_app(lambda: runtime)):
            pass
    runtime.close.assert_awaited_once()


@pytest.mark.parametrize("response", [httpx.Response(503), httpx.Response(429), httpx.Response(200, text="invalid"), httpx.Response(200, json=[])])
def test_auth_upstream_failures_do_not_sign_out_users(response):
    client = SupabaseAuthClient(
        base_url="https://auth.example",
        publishable_key="test-public-key",
        http_client=SimpleNamespace(get=lambda *a, **kw: response),
    )
    with pytest.raises(HTTPException) as caught:
        asyncio.run(client.get_user("token"))
    assert caught.value.status_code == 503


def test_explicit_bearer_token_precedes_stale_cookie():
    request = Request({"type": "http", "headers": [(b"authorization", b"Bearer fresh"),
                      (b"cookie", b"easyteaching_access_token=stale")]})
    assert _request_token(request) == "fresh"


def test_foreign_class_cannot_be_used_to_create_session():
    runtime = ReadyRuntime()
    runtime.store.list_teacher_classes = AsyncMock(return_value=[{"class_id": "own", "name": "Own"}])
    runtime.store.create_conversation_session = AsyncMock()
    app = create_app(lambda: runtime)
    app.dependency_overrides[get_current_user] = lambda: CurrentUser("teacher")
    with TestClient(app) as client:
        assert client.post("/sessions", json={"class_id": "foreign"}).status_code == 403
        assert client.get("/auth/classes").json()["classes"][0]["class_id"] == "own"
    runtime.store.create_conversation_session.assert_not_called()


def test_class_list_requires_active_same_centre_membership():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all([CentreRecord(centre_id="c", name="Centre", state="NSW", suburb="Sydney"),
                    TeacherRecord(teacher_id="t", centre_id="c", display_name="Teacher"),
                    ClassRecord(class_id="allowed", centre_id="c", name="Room", age_group="3-5"),
                    ClassRecord(class_id="inactive", centre_id="c", name="Old", age_group="3-5", active=False),
                    ClassRecord(class_id="foreign", centre_id="other", name="Other", age_group="3-5")])
        for class_id in ("allowed", "inactive", "foreign"):
            db.add(TeacherClassMembershipRecord(membership_id=class_id, teacher_id="t", class_id=class_id))
        db.commit()
    store = object.__new__(AsyncEasyTeachingStore)
    store.session_factory = lambda: AsyncSessionShim(engine)
    assert asyncio.run(store.list_teacher_classes("t")) == [{"class_id": "allowed", "name": "Room"}]
    assert asyncio.run(store.list_teacher_classes("unknown")) == []
    with Session(engine) as db:
        db.get(TeacherRecord, "t").active = False
        db.commit()
    assert asyncio.run(store.list_teacher_classes("t")) == []

    async def denied():
        async with store.session_factory() as session:
            await store._require_class_access(session, "t", "allowed")
    with pytest.raises(ValueError, match="centre scope"):
        asyncio.run(denied())
    engine.dispose()


def test_provisioning_is_idempotent_and_refuses_cross_centre_reassignment():
    from scripts.provision_workspace import provision
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    args = SimpleNamespace(user_id="00000000-0000-0000-0000-000000000001", teacher_name="Teacher",
                           centre_id="centre", centre_name="Centre", state="NSW", suburb="Sydney",
                           class_id="class", class_name="Class", age_group="3-5")
    with Session(engine) as db, db.begin():
        provision(db, args)
        provision(db, args)
    with Session(engine) as db:
        assert db.query(TeacherClassMembershipRecord).count() == 1
    args.centre_id = "other"
    with Session(engine) as db, db.begin():
        with pytest.raises(ValueError, match="another centre"):
            provision(db, args)
        db.rollback()
    engine.dispose()


def test_short_uploaded_pdf_preserves_observation_text(tmp_path):
    from reportlab.pdfgen.canvas import Canvas
    from app.services.file_assets import LocalUploadedFileStore
    from app.services.document_reader import UploadedDocumentReader
    pdf = tmp_path / "observation.pdf"
    canvas = Canvas(str(pdf))
    canvas.drawString(50, 750, "A child builds a bridge with blocks.")
    canvas.save()
    record = LocalUploadedFileStore(tmp_path / "uploads").save_bytes(
        filename=pdf.name, content_type="application/pdf", content=pdf.read_bytes(),
        teacher_id="teacher", class_id="class", session_id="session")
    result = UploadedDocumentReader().read(record)
    assert any("A child builds a bridge" in section.text for section in result.sections)
