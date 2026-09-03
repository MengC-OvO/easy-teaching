"""Replay ordered conversation events over Server-Sent Events."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional, Union

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.dependencies import get_runtime
from app.api.auth import CurrentUser, get_current_user, require_session_owner
from app.schemas import ApiErrorDetail, ApiErrorResponse, RunStatus, StreamEvent
from app.config import settings

if TYPE_CHECKING:
    from app.api.runtime import ApiRuntime


router = APIRouter(prefix="/sessions", tags=["events"])
logger = logging.getLogger(__name__)
_REDIS_EVENT_ID = re.compile(r"^\d+-\d+$")

_STREAM_END_STATUSES = {
    RunStatus.WAITING_FOR_APPROVAL.value,
    RunStatus.COMPLETED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
}


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    session_id: str,
    request_id: str,
) -> JSONResponse:
    error = ApiErrorResponse(
        request_id=request_id,
        error=ApiErrorDetail(
            code=code,
            message=message,
            recoverable=False,
            details={"session_id": session_id},
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(mode="json"),
    )


def _sse_frame(event: StreamEvent) -> str:
    data = json.dumps(
        event.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"id: {event.event_id}\nevent: {event.event.value}\ndata: {data}\n\n"


def _public_event(record: Dict[str, Any]) -> StreamEvent:
    return StreamEvent(
        event_id=record["event_id"],
        event=record["event"],
        sequence=record["sequence"],
        session_id=record["session_id"],
        request_id=record["request_id"],
        data=record["data"],
    )


async def _database_event_stream(
    *,
    runtime: ApiRuntime,
    request: Request,
    request_id: str,
    after_sequence: int,
) -> AsyncIterator[str]:
    cursor = after_sequence
    idle_polls = 0
    while True:
        records = await runtime.store.list_conversation_events(
            request_id=request_id,
            after_sequence=cursor,
        )
        for record in records:
            event = _public_event(record)
            cursor = event.sequence
            idle_polls = 0
            yield _sse_frame(event)

        run = await runtime.store.get_conversation_run(request_id)
        if run is None or run["status"] in _STREAM_END_STATUSES:
            return
        if await request.is_disconnected():
            return

        idle_polls += 1
        if idle_polls >= 150:
            yield ": heartbeat\n\n"
            idle_polls = 0
        # Degraded/inline mode only. Production blocks on Redis Streams.
        await asyncio.sleep(1.0)


async def _redis_event_stream(
    *,
    runtime: ApiRuntime,
    request: Request,
    request_id: str,
    after_event_id: str,
    after_sequence: int,
) -> AsyncIterator[str]:
    cursor = after_event_id
    while True:
        if await request.is_disconnected():
            return
        try:
            records = await runtime.event_bus.read(
                request_id,
                after_event_id=cursor,
                block_ms=settings.redis_progress_block_ms,
                count=50,
            )
        except Exception:
            # Redis progress is disposable. Fall back to low-frequency durable
            # lifecycle polling so a user can still receive the final result.
            logger.warning(
                "Redis progress read failed; using database fallback",
                extra={"request_id": request_id},
                exc_info=True,
            )
            async for frame in _database_event_stream(
                runtime=runtime,
                request=request,
                request_id=request_id,
                after_sequence=after_sequence,
            ):
                yield frame
            return

        for record in records:
            cursor = record.event_id
            event = StreamEvent(
                event_id=record.event_id,
                event=record.event,
                sequence=record.sequence,
                session_id=record.session_id,
                request_id=record.request_id,
                data=record.data,
            )
            yield _sse_frame(event)
            if record.event in {
                "approval_required",
                "completed",
                "failed",
                "cancelled",
            }:
                return

        if records:
            continue
        run = await runtime.store.get_conversation_run(request_id)
        if run is None or run["status"] in _STREAM_END_STATUSES:
            # The terminal Redis notification may have expired or failed. Replay
            # the durable lifecycle record once before closing the connection.
            durable = await runtime.store.list_conversation_events(
                request_id=request_id,
                after_sequence=after_sequence,
            )
            for item in durable:
                if item["event"] in {
                    "approval_required",
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    yield _sse_frame(_public_event(item))
            return
        yield ": heartbeat\n\n"


@router.get(
    "/{session_id}/events",
    response_model=None,
    responses={
        status.HTTP_200_OK: {"content": {"text/event-stream": {}}},
        status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
    },
)
async def stream_events(
    session_id: str,
    request: Request,
    request_id: str = Query(..., min_length=1, max_length=128),
    after_sequence: int = Query(-1, ge=-1),
    after_event_id: Optional[str] = Query(default=None, max_length=64),
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> Union[StreamingResponse, JSONResponse]:
    runtime = get_runtime(request)
    conversation = await runtime.store.get_conversation_session(session_id)
    if conversation is None:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message="The requested session does not exist.",
            session_id=session_id,
            request_id=request_id,
        )
    require_session_owner(conversation, current_user)

    run = await runtime.store.get_conversation_run(request_id)
    if run is None or run["session_id"] != session_id:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="request_not_found",
            message="The requested run does not exist in this session.",
            session_id=session_id,
            request_id=request_id,
        )

    requested_event_id = after_event_id or request.headers.get("Last-Event-ID") or "0-0"
    if not _REDIS_EVENT_ID.fullmatch(requested_event_id):
        requested_event_id = "0-0"
    event_bus = getattr(runtime, "event_bus", None)
    stream = (
        _redis_event_stream(
            runtime=runtime,
            request=request,
            request_id=request_id,
            after_event_id=requested_event_id,
            after_sequence=after_sequence,
        )
        if event_bus is not None and run["status"] not in _STREAM_END_STATUSES
        else _database_event_stream(
            runtime=runtime,
            request=request,
            request_id=request_id,
            after_sequence=after_sequence,
        )
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
