import asyncio
from types import SimpleNamespace

from app.config import settings
from app.schemas import RunStatus
from app.services.redis_rate_limit import RedisRateLimiter
from app.tasks import dispatcher
from app.tasks.celery_app import celery_app
from app.tasks import worker


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.last_key = None

    async def eval(self, script, key_count, key, window):
        assert "INCR" in script
        assert key_count == 1
        self.last_key = key
        self.values[key] = self.values.get(key, 0) + 1
        return [self.values[key], window]


def test_redis_rate_limit_is_shared_by_identity_and_returns_retry_after() -> None:
    redis = FakeRedis()
    limiter = RedisRateLimiter(redis, limit=2, window_seconds=60)

    first = asyncio.run(limiter.check("teacher-1"))
    second = asyncio.run(limiter.check("teacher-1"))
    blocked = asyncio.run(limiter.check("teacher-1"))
    other_teacher = asyncio.run(limiter.check("teacher-2"))

    assert first.allowed and first.remaining == 1
    assert second.allowed and second.remaining == 0
    assert not blocked.allowed and blocked.retry_after_seconds == 60
    assert other_teacher.allowed
    assert "teacher-2" not in redis.last_key


class FakeOutboxStore:
    def __init__(self):
        self.status = "pending"
        self.publish_attempts = 0
        self.celery_task_id = None
        self.error = None

    async def claim_conversation_task_for_publish(self, request_id, *, lease_seconds):
        if self.status != "pending":
            return None
        self.status = "publishing"
        self.publish_attempts += 1
        return {"request_id": request_id, "publish_attempts": self.publish_attempts}

    async def finish_conversation_task_publish(
        self, request_id, *, celery_task_id=None, error=None, retry_delay_seconds=0
    ):
        self.status = "published" if error is None else "pending"
        self.celery_task_id = celery_task_id
        self.error = error

    async def list_publishable_conversation_task_ids(self, *, limit):
        return ["request-1"] if self.status == "pending" else []


def test_outbox_publish_failure_stays_durable_and_next_attempt_succeeds(monkeypatch) -> None:
    store = FakeOutboxStore()
    calls = 0

    def send_task(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("synthetic Redis outage")
        return SimpleNamespace(id=kwargs["task_id"])

    monkeypatch.setattr(dispatcher.celery_app, "send_task", send_task)

    assert asyncio.run(dispatcher.publish_outbox_task(store, "request-1")) is False
    assert store.status == "pending"
    assert "ConnectionError" in store.error
    assert asyncio.run(dispatcher.publish_outbox_task(store, "request-1")) is True
    assert store.status == "published"
    assert store.celery_task_id == "request-1"


def test_concurrent_publish_attempts_send_only_one_broker_message(monkeypatch) -> None:
    store = FakeOutboxStore()
    sent = []

    def send_task(*args, **kwargs):
        sent.append(kwargs["task_id"])
        return SimpleNamespace(id=kwargs["task_id"])

    monkeypatch.setattr(dispatcher.celery_app, "send_task", send_task)

    async def publish_concurrently():
        return await asyncio.gather(
            *(
                dispatcher.publish_outbox_task(store, "request-1")
                for _ in range(100)
            )
        )

    results = asyncio.run(publish_concurrently())

    assert results.count(True) == 1
    assert results.count(False) == 99
    assert sent == ["request-1"]


def test_celery_configuration_is_safe_for_long_agent_tasks() -> None:
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_ignore_result is True
    assert celery_app.conf.accept_content == ["json"]
    assert (
        celery_app.conf.broker_transport_options["visibility_timeout"]
        > settings.celery_task_time_limit_seconds
    )


class FakeWorkerStore:
    def __init__(self):
        self.claimed = True
        self.finished = []

    async def claim_conversation_task_for_execution(self, request_id, *, lease_seconds):
        if not self.claimed:
            return None
        self.claimed = False
        return {
            "request_id": request_id,
            "execution_attempts": 1,
            "payload": {
                "request_id": request_id,
                "session_id": "session-1",
                "thread_id": "thread-1",
                "teacher_id": "teacher-1",
                "class_id": "class-1",
                "message": "redacted message",
                "privacy_mapping_id": None,
            },
        }

    async def get_conversation_run(self, request_id):
        return {"request_id": request_id, "session_id": "session-1", "status": "accepted"}

    async def finish_conversation_task_execution(self, request_id, *, status, error=None):
        self.finished.append((request_id, status, error))


class EmptyCheckpointGraph:
    async def aget_state(self, config):
        return SimpleNamespace(values={}, next=())


def test_worker_claim_prevents_duplicate_model_execution(monkeypatch) -> None:
    store = FakeWorkerStore()
    runtime = SimpleNamespace(store=store, graph=EmptyCheckpointGraph())
    calls = []

    async def execute(**kwargs):
        calls.append(kwargs)
        return RunStatus.COMPLETED

    monkeypatch.setattr(worker, "execute_message", execute)

    async def execute_concurrently():
        return await asyncio.gather(
            *(worker._execute(runtime, "request-1") for _ in range(100))
        )

    results = asyncio.run(execute_concurrently())

    assert results.count("completed") == 1
    assert results.count("duplicate_or_finished") == 99
    assert len(calls) == 1
    assert calls[0]["message"] == "redacted message"
    assert store.finished == [("request-1", "completed", None)]


def test_recoverable_worker_error_releases_execution_lease_for_retry() -> None:
    store = FakeWorkerStore()
    runtime = SimpleNamespace(store=store, graph=EmptyCheckpointGraph())

    asyncio.run(
        worker._prepare_retry(
            runtime,
            "request-1",
            TimeoutError("synthetic provider timeout"),
            exhausted=False,
        )
    )

    assert store.finished == [
        (
            "request-1",
            "published",
            "TimeoutError: synthetic provider timeout",
        )
    ]
