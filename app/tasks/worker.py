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
from app.api.recovery_state import checkpoint_belongs_to_run, completed_snapshot_status
from app.config import settings
from app.schemas import GraphState, RunStatus
from app.tasks.celery_app import celery_app
from app.services.task_lease import LeaseLostError, bind_execution_lease


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

    token = task["lease_token"]
    with bind_execution_lease(request_id, token):
        try:
            return await _execute_claimed(runtime, request_id, task)
        except Exception as error:
            error.execution_lease_token = token
            raise


async def _execute_claimed(runtime: ApiRuntime, request_id: str, task: dict) -> str:
    token = task["lease_token"]

    if task["payload"].get("kind") == "approved_action":
        if task.get("execution_attempts", 1) > settings.celery_task_max_retries + 1:
            await runtime.store.fail_approved_action(action_id=task["payload"]["action_id"],
                message="Approved action exhausted its durable execution attempts.")
            run = await runtime.store.get_conversation_run(request_id)
            status = run["status"]
        else:
            status = await runtime.store.execute_approved_action(
                action_id=task["payload"]["action_id"], registry=runtime.tool_registry,
            )
        finished = await runtime.store.finish_conversation_task_execution(
            request_id, lease_token=token, status="completed",
        )
        if not finished:
            raise LeaseLostError("Action completion rejected after lease loss")
        # Durable lifecycle events already committed with the action result.
        from app.api.execution import publish_progress_best_effort
        await publish_progress_best_effort(runtime, request_id=request_id,
            session_id=task["payload"]["session_id"], event=status, data={"status": status})
        return status

    run = await runtime.store.get_conversation_run(request_id)
    if run is None:
        await runtime.store.finish_conversation_task_execution(
            request_id, lease_token=token, status="dead", error="Conversation run does not exist"
        )
        return "missing_run"
    if run["status"] in {
        RunStatus.COMPLETED.value,
        RunStatus.FAILED.value,
        RunStatus.WAITING_FOR_APPROVAL.value,
    }:
        await runtime.store.finish_conversation_task_execution(request_id, lease_token=token, status="completed")
        return run["status"]

    payload = task["payload"]
    if payload["session_id"] != run["session_id"]:
        raise ValueError("Conversation task belongs to another session")
    # Attempts count claims, not graph progress. Use the owned checkpoint to
    # distinguish resuming nodes from finalizing an already-finished graph.
    snapshot = await runtime.graph.aget_state(checkpoint_config(payload["thread_id"]))
    owned = checkpoint_belongs_to_run(
        snapshot.values, request_id=request_id, session_id=run["session_id"]
    )
    if owned and not snapshot.next:
        state = GraphState.model_validate(snapshot.values)
        status = await persist_run_outcome(
            runtime=runtime,
            request_id=request_id,
            session_id=run["session_id"],
            state=state,
            final_status=completed_snapshot_status(state),
        )
    elif owned:
        status = await execute_checkpoint_resume(
            runtime=runtime,
            request_id=request_id,
            session_id=payload["session_id"],
            thread_id=payload["thread_id"],
            propagate_incomplete_error=True,
        )
    else:
        # A new accepted turn may inherit a finished previous turn's context.
        # Never overwrite a foreign pending graph or restart a running run
        # whose own checkpoint is missing.
        if run["status"] != RunStatus.ACCEPTED.value:
            raise ValueError("Running conversation has no matching recovery checkpoint")
        if snapshot.values and (
            snapshot.values.get("session_id") != run["session_id"] or snapshot.next
        ):
            raise ValueError("Recovery checkpoint belongs to another active run or session")
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
    finished = await runtime.store.finish_conversation_task_execution(request_id, lease_token=token, status="completed")
    if not finished:
        raise LeaseLostError("Execution completion rejected after lease loss")
    return status.value


async def _prepare_retry(
    runtime: ApiRuntime, request_id: str, error: BaseException, *, exhausted: bool,
    lease_token: str,
) -> None:
    with bind_execution_lease(request_id, lease_token):
        await _prepare_owned_retry(runtime, request_id, error, exhausted=exhausted, lease_token=lease_token)


async def _prepare_owned_retry(
    runtime: ApiRuntime, request_id: str, error: BaseException, *, exhausted: bool,
    lease_token: str,
) -> None:
    if not exhausted:
        # Return to a claimable state before Celery schedules the retry message.
        released = await runtime.store.finish_conversation_task_execution(
            request_id,
            lease_token=lease_token,
            status="published",
            error=f"{type(error).__name__}: {error}",
        )
        if not released:
            raise LeaseLostError("Retry release rejected after lease loss")
        return
    task = await runtime.store.get_conversation_task(request_id)
    if task is not None and task["payload"].get("kind") == "approved_action":
        await runtime.store.fail_approved_action(action_id=task["payload"]["action_id"],
            message="Approved action could not be completed after bounded retries.")
        await runtime.store.finish_conversation_task_execution(
            request_id, lease_token=lease_token, status="dead", error="Approved action retries exhausted",
        )
        return
    state = None
    if task is not None:
        try:
            snapshot = await runtime.graph.aget_state(
                checkpoint_config(task["payload"]["thread_id"])
            )
            if checkpoint_belongs_to_run(
                snapshot.values,
                request_id=request_id,
                session_id=task["payload"]["session_id"],
            ):
                state = GraphState.model_validate(snapshot.values)
        except Exception:
            state = None
    run = await runtime.store.get_conversation_run(request_id)
    if run is not None:
        if state is not None and state.session_id != run["session_id"]:
            state = None
        await persist_run_outcome(
            runtime=runtime,
            request_id=request_id,
            session_id=run["session_id"],
            state=state,
            final_status=RunStatus.FAILED,
        )
    await runtime.store.finish_conversation_task_execution(
        request_id,
        lease_token=lease_token,
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
    except LeaseLostError:
        return "lease_lost"
    except Exception as error:
        exhausted = self.request.retries >= settings.celery_task_max_retries
        token = getattr(error, "execution_lease_token", None)
        if token is not None:
            try:
                loop.run_until_complete(
                    _prepare_retry(runtime, request_id, error, exhausted=exhausted, lease_token=token)
                )
            except LeaseLostError:
                return "lease_lost"
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
