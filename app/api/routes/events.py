"""Replay ordered conversation events over Server-Sent Events."""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional, Union

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.dependencies import get_runtime
from app.api.auth import CurrentUser, get_current_user, require_session_owner
from app.api.runtime import ApiRuntime
from app.schemas import ApiErrorDetail, ApiErrorResponse, RunStatus, StreamEvent


router = APIRouter(prefix="/sessions", tags=["events"])

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


async def _event_stream(
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
        await asyncio.sleep(0.1)


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

    return StreamingResponse(
        _event_stream(
            runtime=runtime,
            request=request,
            request_id=request_id,
            after_sequence=after_sequence,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
