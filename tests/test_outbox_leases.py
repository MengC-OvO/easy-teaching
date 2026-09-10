"""Exercise real SQL predicates with SQLite; PostgreSQL row locks need live tests."""
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from app.services.async_store import AsyncEasyTeachingStore
from app.services.models import Base, ConversationRunRecord, ConversationTaskOutboxRecord
from app.services.task_lease import LeaseLostError, bind_execution_lease
from app.tasks import dispatcher, worker


class AsyncSessionShim:
    def __init__(self, engine):
        self.session = Session(engine, expire_on_commit=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.session.close()

    async def execute(self, statement):
        return self.session.execute(statement)

    async def get(self, model, key):
        return self.session.get(model, key)

    def add(self, record):
        self.session.add(record)

    async def commit(self):
        self.session.commit()


@pytest.fixture
def store():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(ConversationRunRecord(request_id="req", session_id="session", status="accepted"))
        db.add(ConversationTaskOutboxRecord(request_id="req", status="pending", payload={}))
        db.commit()
    value = object.__new__(AsyncEasyTeachingStore)
    value.session_factory = lambda: AsyncSessionShim(engine)
    value.test_engine = engine
    yield value
    engine.dispose()


def change(store, **values):
    with Session(store.test_engine) as db:
        db.execute(update(ConversationTaskOutboxRecord).where(
            ConversationTaskOutboxRecord.request_id == "req"
        ).values(**values))
        db.commit()


async def publish(store):
    claim = await store.claim_conversation_task_for_publish("req", lease_seconds=30)
    assert await store.finish_conversation_task_publish("req", lease_token=claim["lease_token"], celery_task_id="req")
    return claim


def test_early_worker_claim_fences_late_publisher_success_and_failure(store):
    async def scenario():
        publisher = await store.claim_conversation_task_for_publish("req", lease_seconds=30)
        executor = await store.claim_conversation_task_for_execution("req", lease_seconds=60)
        assert executor and executor["lease_token"] != publisher["lease_token"]
        for error in (None, "uncertain broker response"):
            assert not await store.finish_conversation_task_publish(
                "req", lease_token=publisher["lease_token"], error=error,
            )
        assert (await store.get_conversation_task("req"))["status"] == "running"
        assert await store.claim_conversation_task_for_execution("req", lease_seconds=60) is None
    asyncio.run(scenario())


def test_expired_publisher_cannot_finish_after_takeover(store):
    async def scenario():
        old = await store.claim_conversation_task_for_publish("req", lease_seconds=30)
        change(store, lease_until=datetime.utcnow() - timedelta(seconds=1))
        new = await store.claim_conversation_task_for_publish("req", lease_seconds=30)
        assert new["lease_token"] != old["lease_token"]
        assert not await store.finish_conversation_task_publish("req", lease_token=old["lease_token"])
        assert await store.finish_conversation_task_publish("req", lease_token=new["lease_token"])
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["completed", "dead", "published"])
def test_old_executor_cannot_complete_fail_or_release_new_lease(store, status):
    async def scenario():
        await publish(store)
        old = await store.claim_conversation_task_for_execution("req", lease_seconds=60)
        change(store, lease_until=datetime.utcnow() - timedelta(seconds=1))
        new = await store.claim_conversation_task_for_execution("req", lease_seconds=60)
        assert not await store.finish_conversation_task_execution("req", lease_token=old["lease_token"], status=status)
        current = await store.get_conversation_task("req")
        assert current["lease_token"] == new["lease_token"]
        assert current["status"] == "running"
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["lost_published", "expired_running", "expired_publishing"])
def test_reconciler_requeues_stranded_tasks_once(store, kind):
    async def scenario():
        await publish(store)
        if kind == "lost_published":
            change(store, updated_at=datetime.utcnow() - timedelta(seconds=1201))
        else:
            change(store, status=kind.removeprefix("expired_"), lease_token="old",
                   lease_until=datetime.utcnow() - timedelta(seconds=1))
        assert await store.reconcile_stalled_conversation_tasks(stale_seconds=1200) == 1
        task = await store.get_conversation_task("req")
        assert task["status"] == "pending" and task["lease_token"] is None
        assert await store.reconcile_stalled_conversation_tasks(stale_seconds=1200) == 0
        assert await store.claim_conversation_task_for_publish("req", lease_seconds=30)
    asyncio.run(scenario())


def test_reconciler_leaves_fresh_queue_and_live_executor_alone(store):
    async def scenario():
        await publish(store)
        assert await store.reconcile_stalled_conversation_tasks(stale_seconds=1200) == 0
        await store.claim_conversation_task_for_execution("req", lease_seconds=60)
        assert await store.reconcile_stalled_conversation_tasks(stale_seconds=1200) == 0
    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["completed", "failed", "waiting_for_approval", "cancelled"])
def test_reconciler_does_not_republish_terminal_runs(store, status):
    async def scenario():
        await publish(store)
        change(store, updated_at=datetime.utcnow() - timedelta(seconds=1201))
        await store.update_conversation_run_status("req", status)
        await store.reconcile_stalled_conversation_tasks(stale_seconds=1200)
        assert (await store.get_conversation_task("req"))["status"] == "completed"
    asyncio.run(scenario())


def test_stale_owner_cannot_write_result_run_status_or_event(store):
    async def scenario():
        await publish(store)
        old = await store.claim_conversation_task_for_execution("req", lease_seconds=60)
        change(store, lease_until=datetime.utcnow() - timedelta(seconds=1))
        new = await store.claim_conversation_task_for_execution("req", lease_seconds=60)
        with bind_execution_lease("req", old["lease_token"]):
            with pytest.raises(LeaseLostError):
                await store.update_conversation_run_status("req", "failed")
            with pytest.raises(LeaseLostError):
                await store.save_conversation_run_result(request_id="req", session_id="session", draft={}, approval={}, citations=[])
            with pytest.raises(LeaseLostError):
                await store.append_conversation_event(request_id="req", session_id="session", event="failed", data={})
        with bind_execution_lease("req", new["lease_token"]):
            await store.update_conversation_run_status("req", "completed")
        assert (await store.get_conversation_run("req"))["status"] == "completed"
    asyncio.run(scenario())


def test_expired_token_is_rejected_even_before_new_owner_claims(store):
    async def scenario():
        await publish(store)
        old = await store.claim_conversation_task_for_execution("req", lease_seconds=60)
        change(store, lease_until=datetime.utcnow() - timedelta(seconds=1))
        assert not await store.finish_conversation_task_execution("req", lease_token=old["lease_token"], status="completed")
    asyncio.run(scenario())


def test_relay_runs_compensation_before_publishing(monkeypatch):
    calls = []
    async def reconcile(**kwargs):
        calls.append("reconcile")
    async def pending(**kwargs):
        calls.append("list")
        return ["req"]
    async def send(*args):
        calls.append("publish")
        return True
    fake = SimpleNamespace(reconcile_stalled_conversation_tasks=reconcile, list_publishable_conversation_task_ids=pending)
    monkeypatch.setattr(dispatcher, "publish_outbox_task", send)
    assert asyncio.run(dispatcher.OutboxRelay(fake).relay_once()) == 1
    assert calls == ["reconcile", "list", "publish"]


def test_retry_cannot_release_another_execution_token():
    fake = SimpleNamespace(store=SimpleNamespace(finish_conversation_task_execution=AsyncMock(return_value=False)))
    with pytest.raises(LeaseLostError):
        asyncio.run(worker._prepare_retry(fake, "req", RuntimeError("synthetic"), exhausted=False, lease_token="old"))
