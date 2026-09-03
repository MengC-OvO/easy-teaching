"""Celery worker entry point for durable LangGraph runs."""

from __future__ import annotations

import asyncio
import random
from typing import Optional

from celery import signals
from app.api.checkpoint_config import checkpoint_config
from app.api.execution import (
    execute_checkpoint_resume,
    execute_message,
    persist_run_outcome,
)
from app.api.runtime import ApiRuntime, build_api_runtime
from app.config import settings
from app.schemas import GraphState, RunStatus
from app.tasks.celery_app import celery_app


_loop: Optional[asyncio.AbstractEventLoop] = None
_runtime: Optional[ApiRuntime] = None


def _worker_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None:
        _loop = asyncio.new_event_loop()
    return _loop


def _worker_runtime() -> ApiRuntime:
    global _runtime
    if _runtime is None:
        _runtime = _worker_loop().run_until_complete(build_api_runtime())
    return _runtime


async def _execute(runtime: ApiRuntime, request_id: str) -> str:
    task = await runtime.store.claim_conversation_task_for_execution(
        request_id,
        lease_seconds=settings.celery_task_time_limit_seconds + 120,
    )
    if task is None:
        return "duplicate_or_finished"

    run = await runtime.store.get_conversation_run(request_id)
    if run is None:
        await runtime.store.finish_conversation_task_execution(
            request_id, status="dead", error="Conversation run does not exist"
        )
        return "missing_run"
    if run["status"] in {
        RunStatus.COMPLETED.value,
        RunStatus.FAILED.value,
        RunStatus.WAITING_FOR_APPROVAL.value,
    }:
        await runtime.store.finish_conversation_task_execution(request_id, status="completed")
        return run["status"]

    payload = task["payload"]
    # A retry resumes a durable checkpoint when possible instead of blindly
    # starting the model call from the beginning.
    snapshot = await runtime.graph.aget_state(checkpoint_config(payload["thread_id"]))
    if task["execution_attempts"] > 1 and snapshot.values and snapshot.next:
        status = await execute_checkpoint_resume(
            runtime=runtime,
            request_id=request_id,
            session_id=payload["session_id"],
            thread_id=payload["thread_id"],
            propagate_incomplete_error=True,
        )
    else:
        status = await execute_message(
            runtime=runtime,
            request_id=request_id,
            session_id=payload["session_id"],
            thread_id=payload["thread_id"],
            teacher_id=payload.get("teacher_id"),
            class_id=payload.get("class_id"),
            message=payload["message"],
            privacy_mapping_id=payload.get("privacy_mapping_id"),
            propagate_incomplete_error=True,
        )
    await runtime.store.finish_conversation_task_execution(request_id, status="completed")
    return status.value


async def _prepare_retry(
    runtime: ApiRuntime, request_id: str, error: BaseException, *, exhausted: bool
) -> None:
    if not exhausted:
        # Return to a claimable state before Celery schedules the retry message.
        await runtime.store.finish_conversation_task_execution(
            request_id,
            status="published",
            error=f"{type(error).__name__}: {error}",
        )
        return
    task = await runtime.store.get_conversation_task(request_id)
    state = None
    if task is not None:
        try:
            snapshot = await runtime.graph.aget_state(
                checkpoint_config(task["payload"]["thread_id"])
            )
            if snapshot.values:
                state = GraphState.model_validate(snapshot.values)
        except Exception:
            state = None
    run = await runtime.store.get_conversation_run(request_id)
    if run is not None:
        await persist_run_outcome(
            runtime=runtime,
            request_id=request_id,
            session_id=run["session_id"],
            state=state,
            final_status=RunStatus.FAILED,
        )
    await runtime.store.finish_conversation_task_execution(
        request_id,
        status="dead",
        error=f"{type(error).__name__}: {error}",
    )


@celery_app.task(
    bind=True,
    name="easyteaching.execute_conversation",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=settings.celery_task_max_retries,
)
def execute_conversation(self, request_id: str) -> str:
    loop = _worker_loop()
    runtime = _worker_runtime()
    try:
        return loop.run_until_complete(_execute(runtime, request_id))
    except Exception as error:
        exhausted = self.request.retries >= settings.celery_task_max_retries
        loop.run_until_complete(
            _prepare_retry(runtime, request_id, error, exhausted=exhausted)
        )
        if exhausted:
            raise
        base = min(60, 2 ** (self.request.retries + 1))
        raise self.retry(exc=error, countdown=base + random.uniform(0, base * 0.25))


@signals.worker_process_shutdown.connect
def close_worker_runtime(**_: object) -> None:
    global _runtime, _loop
    if _runtime is not None and _loop is not None:
        _loop.run_until_complete(_runtime.close())
    if _loop is not None:
        _loop.close()
    _runtime = None
    _loop = None
