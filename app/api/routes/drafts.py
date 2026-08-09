"""Retrieve request-scoped draft snapshots produced by EduFlow runs."""

from typing import Optional, Union

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_runtime
from app.api.auth import CurrentUser, get_current_user, require_session_owner
from app.schemas import ApiErrorDetail, ApiErrorResponse, DraftResponse, RunStatus


router = APIRouter(prefix="/sessions", tags=["drafts"])


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    session_id: str,
    request_id: str,
    recoverable: bool,
) -> JSONResponse:
    error = ApiErrorResponse(
        request_id=request_id,
        error=ApiErrorDetail(
            code=code,
            message=message,
            recoverable=recoverable,
            details={"session_id": session_id},
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(mode="json"),
    )


@router.get(
    "/{session_id}/drafts/{request_id}",
    response_model=DraftResponse,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ApiErrorResponse},
    },
)
async def get_draft(
    session_id: str,
    request_id: str,
    request: Request,
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> Union[DraftResponse, JSONResponse]:
    runtime = get_runtime(request)
    conversation = await runtime.store.get_conversation_session(session_id)
    if conversation is None:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message="The requested session does not exist.",
            session_id=session_id,
            request_id=request_id,
            recoverable=False,
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
            recoverable=False,
        )

    result = await runtime.store.get_conversation_run_result(request_id)
    if result is None:
        run_status = RunStatus(run["status"])
        in_progress = run_status in {RunStatus.ACCEPTED, RunStatus.RUNNING}
        return _error_response(
            status_code=(
                status.HTTP_409_CONFLICT
                if in_progress
                else status.HTTP_404_NOT_FOUND
            ),
            code="draft_not_ready" if in_progress else "draft_not_found",
            message=(
                "The draft is not ready yet."
                if in_progress
                else "The run did not produce a draft."
            ),
            session_id=session_id,
            request_id=request_id,
            recoverable=in_progress,
        )

    return DraftResponse(
        session_id=session_id,
        request_id=request_id,
        status=run["status"],
        draft=result["draft"],
        approval=result["approval"],
        citations=result["citations"],
    )
