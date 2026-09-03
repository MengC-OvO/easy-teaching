"""Opt-in tests against real PostgreSQL and Redis services."""

import asyncio
import os
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from sqlalchemy import text

from app.asyncio_compat import run_async
from app.api.errors import ConversationSessionBusyError
from app.config import settings
from app.services.async_store import AsyncEasyTeachingStore
from app.services.redis_event_bus import RedisEventBus
from app.tasks.dispatcher import publish_outbox_task


DATABASE_URL = os.getenv("INTEGRATION_DATABASE_URL", "")
REDIS_URL = os.getenv("INTEGRATION_REDIS_URL", "")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL or not REDIS_URL,
    reason="Set INTEGRATION_DATABASE_URL and INTEGRATION_REDIS_URL",
)


async def _cleanup(store, *, session_id, request_ids):
    async with store.engine.begin() as connection:
        await connection.execute(
            text("DELETE FROM conversation_events WHERE request_id = ANY(:ids)"),
            {"ids": request_ids},
        )
        await connection.execute(
            text("DELETE FROM conversation_task_outbox WHERE request_id = ANY(:ids)"),
            {"ids": request_ids},
        )
        await connection.execute(
            text("DELETE FROM conversation_runs WHERE request_id = ANY(:ids)"),
            {"ids": request_ids},
        )
        await connection.execute(
            text("DELETE FROM conversation_sessions WHERE session_id = :session_id"),
            {"session_id": session_id},
        )


def test_real_postgres_commits_run_and_outbox_atomically_and_enforces_session_busy():
    async def scenario():
        store = AsyncEasyTeachingStore(DATABASE_URL)
        session_id = f"it-session-{uuid4()}"
        request_id = f"it-request-{uuid4()}"
        second_request_id = f"it-request-{uuid4()}"
        try:
            await store.initialize()
            await store.create_conversation_session(
                session_id=session_id,
                thread_id=f"it-thread-{uuid4()}",
                teacher_id=None,
                class_id=None,
            )
            payload = {
                "request_id": request_id,
                "session_id": session_id,
                "thread_id": "integration-thread",
                "message": "redacted integration request",
            }
            run = await store.create_conversation_run(
                request_id=request_id,
                session_id=session_id,
                task_payload=payload,
            )
            task = await store.get_conversation_task(request_id)
            assert run["created"] is True
            assert task is not None
            assert task["status"] == "pending"
            assert task["payload"] == payload

            with pytest.raises(ConversationSessionBusyError):
                await store.create_conversation_run(
                    request_id=second_request_id,
                    session_id=session_id,
                    task_payload={"request_id": second_request_id},
                )
            assert await store.get_conversation_run(second_request_id) is None
            assert await store.get_conversation_task(second_request_id) is None
        finally:
            await _cleanup(
                store,
                session_id=session_id,
                request_ids=[request_id, second_request_id],
            )
            await store.close()

    run_async(scenario())


def test_real_postgres_rolls_back_run_when_outbox_payload_cannot_be_serialized():
    async def scenario():
        store = AsyncEasyTeachingStore(DATABASE_URL)
        session_id = f"it-session-{uuid4()}"
        request_id = f"it-request-{uuid4()}"
        try:
            await store.initialize()
            await store.create_conversation_session(
                session_id=session_id,
                thread_id=f"it-thread-{uuid4()}",
                teacher_id=None,
                class_id=None,
            )
            with pytest.raises(Exception):
                await store.create_conversation_run(
                    request_id=request_id,
                    session_id=session_id,
                    task_payload={"not_json": object()},
                )
            assert await store.get_conversation_run(request_id) is None
            assert await store.get_conversation_task(request_id) is None
        finally:
            await _cleanup(store, session_id=session_id, request_ids=[request_id])
            await store.close()

    run_async(scenario())


def test_real_redis_stream_is_ordered_replayable_capped_and_expiring():
    async def scenario():
        client = Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        request_id = f"it-progress-{uuid4()}"
        bus = RedisEventBus(client, maxlen=500, ttl_seconds=120)
        try:
            assert await client.ping()
            await asyncio.gather(
                *(
                    bus.publish(
                        request_id=request_id,
                        session_id="it-session",
                        event="trace",
                        data={"index": index},
                    )
                    for index in range(200)
                )
            )
            events = await bus.read(
                request_id,
                after_event_id="0-0",
                block_ms=1000,
                count=250,
            )
            assert len(events) == 200
            assert [item.sequence for item in events] == list(range(1, 201))
            assert await client.xlen(bus._stream_key(request_id)) == 200
            ttl = await client.ttl(bus._stream_key(request_id))
            assert 0 < ttl <= 120
        finally:
            await client.delete(
                bus._stream_key(request_id),
                bus._sequence_key(request_id),
            )
            await client.aclose()

    run_async(scenario())


def test_real_outbox_relay_publishes_request_id_to_redis_broker():
    async def scenario():
        store = AsyncEasyTeachingStore(DATABASE_URL)
        broker = Redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
        session_id = f"it-session-{uuid4()}"
        request_id = f"it-request-{uuid4()}"
        try:
            await store.initialize()
            await store.create_conversation_session(
                session_id=session_id,
                thread_id=f"it-thread-{uuid4()}",
                teacher_id=None,
                class_id=None,
            )
            await store.create_conversation_run(
                request_id=request_id,
                session_id=session_id,
                task_payload={
                    "request_id": request_id,
                    "session_id": session_id,
                    "thread_id": "it-thread",
                    "message": "redacted integration request",
                },
            )
            assert await publish_outbox_task(store, request_id) is True
            task = await store.get_conversation_task(request_id)
            assert task["status"] == "published"
            assert task["celery_task_id"] == request_id
            assert await broker.llen(settings.celery_queue_name) == 1
        finally:
            await broker.delete(settings.celery_queue_name)
            await broker.aclose()
            await _cleanup(store, session_id=session_id, request_ids=[request_id])
            await store.close()

    run_async(scenario())
