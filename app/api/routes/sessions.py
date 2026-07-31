"""Create and retrieve durable EduFlow conversation sessions."""

from typing import Any, Dict, Union
from uuid import uuid4

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_runtime
from app.schemas import (
    ApiErrorDetail,
    ApiErrorResponse,
    SessionCreateRequest,
    SessionCreateResponse,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


def _public_session(record: Dict[str, Any]) -> SessionCreateResponse:
    return SessionCreateResponse(
        session_id=record["session_id"],
        thread_id=record["thread_id"],
        status=record["status"],
    )


@router.post(
    "",
    response_model=SessionCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    payload: SessionCreateRequest,
    request: Request,
) -> SessionCreateResponse:
    record = get_runtime(request).store.create_conversation_session(
        session_id=str(uuid4()),
        thread_id=str(uuid4()),
        teacher_id=payload.teacher_id,
        class_id=payload.class_id,
    )
    return _public_session(record)


@router.get(
    "/{session_id}",
    response_model=SessionCreateResponse,
    responses={status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse}},
)
def get_session(
    session_id: str,
    request: Request,
) -> Union[SessionCreateResponse, JSONResponse]:
    record = get_runtime(request).store.get_conversation_session(session_id)
    if record is not None:
        return _public_session(record)

    error = ApiErrorResponse(
        error=ApiErrorDetail(
            code="session_not_found",
            message="The requested session does not exist.",
            recoverable=False,
            details={"session_id": session_id},
        )
    )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=error.model_dump(mode="json"),
    )
