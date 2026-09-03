"""Transactional-Outbox relay from PostgreSQL to Celery/Redis."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.config import settings
from app.tasks.celery_app import celery_app


logger = logging.getLogger(__name__)
TASK_NAME = "easyteaching.execute_conversation"


async def publish_outbox_task(store: Any, request_id: str) -> bool:
    """Publish one leased row; leave it retryable if Redis is unavailable."""
    claimed = await store.claim_conversation_task_for_publish(request_id, lease_seconds=30)
    if claimed is None:
        return False
    try:
        result = await asyncio.to_thread(
            celery_app.send_task,
            TASK_NAME,
            args=[request_id],
            task_id=request_id,
            queue=settings.celery_queue_name,
        )
    except Exception as error:
        delay = min(60, 2 ** min(claimed["publish_attempts"], 6))
        await store.finish_conversation_task_publish(
            request_id,
            error=f"{type(error).__name__}: {error}",
            retry_delay_seconds=delay,
        )
        logger.warning("outbox publish failed", extra={"request_id": request_id})
        return False
    await store.finish_conversation_task_publish(
        request_id,
        celery_task_id=result.id,
    )
    return True


class OutboxRelay:
    """Small API-side relay; PostgreSQL leases make multiple API replicas safe."""

    def __init__(self, store: Any, *, poll_seconds: Optional[float] = None) -> None:
        self.store = store
        self.poll_seconds = poll_seconds or settings.outbox_poll_interval_seconds
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="outbox-relay")

    def notify(self) -> None:
        self._wake.set()

    async def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._task is not None:
            await self._task

    async def relay_once(self) -> int:
        request_ids = await self.store.list_publishable_conversation_task_ids(limit=100)
        published = 0
        for request_id in request_ids:
            published += int(await publish_outbox_task(self.store, request_id))
        return published

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.relay_once()
            except Exception:
                logger.exception("outbox relay iteration failed")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass
