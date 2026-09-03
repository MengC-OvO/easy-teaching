from fastapi.testclient import TestClient

from app.main import app, create_app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "easy-teaching",
        "environment": "local",
    }


class ReadyStore:
    async def healthcheck(self):
        return True

    async def list_conversation_runs(self, *, statuses=None):
        return []


class ReadyRedis:
    def __init__(self, ready=True):
        self.ready = ready

    async def ping(self):
        if not self.ready:
            raise ConnectionError("synthetic outage")
        return True


class ReadyRuntime:
    def __init__(self, redis_ready=True):
        self.store = ReadyStore()
        self.redis_client = ReadyRedis(redis_ready)
        self.redis_progress_client = ReadyRedis(redis_ready)

    async def close(self):
        return None


def test_readiness_checks_postgres_and_redis() -> None:
    ready_app = create_app(runtime_factory=lambda: ReadyRuntime())
    with TestClient(ready_app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"] == {
        "postgres": "ok",
        "redis": "ok",
        "redis_progress": "ok",
    }


def test_readiness_fails_when_redis_is_unavailable() -> None:
    ready_app = create_app(runtime_factory=lambda: ReadyRuntime(redis_ready=False))
    with TestClient(ready_app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["redis"] == "unavailable"
    assert response.json()["checks"]["redis_progress"] == "unavailable"
