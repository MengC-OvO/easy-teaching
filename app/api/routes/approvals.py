"""Resume approval-interrupted EduFlow conversation runs."""

from typing import Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_runtime
from app.api.auth import CurrentUser, get_current_user, require_session_owner
from app.api.execution import execute_approval
from app.schemas import (
    ApiErrorDetail,
    ApiErrorResponse,
    ApprovalSubmitRequest,
    ApprovalSubmitResponse,
    RunStatus,
    StreamEventType,
)


router = APIRouter(prefix="/sessions", tags=["approvals"])


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


@router.post(
    "/{session_id}/approvals",
    response_model=ApprovalSubmitResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ApiErrorResponse},
    },
)
def submit_approval(
    session_id: str,
    payload: ApprovalSubmitRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> Union[ApprovalSubmitResponse, JSONResponse]:
    runtime = get_runtime(request)
    conversation = runtime.store.get_conversation_session(session_id)
    if conversation is None:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message="The requested session does not exist.",
            session_id=session_id,
            request_id=payload.request_id,
        )
    require_session_owner(conversation, current_user)

    run = runtime.store.get_conversation_run(payload.request_id)
    if run is None or run["session_id"] != session_id:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="request_not_found",
            message="The requested run does not exist in this session.",
            session_id=session_id,
            request_id=payload.request_id,
        )

    existing = runtime.store.get_approval_decision(payload.request_id)
    if existing is not None:
        if existing["decision"] != payload.decision.value:
            return _error_response(
                status_code=status.HTTP_409_CONFLICT,
                code="approval_decision_conflict",
                message="A different approval decision was already submitted.",
                session_id=session_id,
                request_id=payload.request_id,
            )
        return ApprovalSubmitResponse(
            session_id=session_id,
            request_id=payload.request_id,
            decision=payload.decision,
            status=run["status"],
        )

    if run["status"] != RunStatus.WAITING_FOR_APPROVAL.value:
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="approval_not_pending",
            message="The request is not waiting for approval.",
            session_id=session_id,
            request_id=payload.request_id,
        )

    approval = runtime.store.create_approval_decision(
        request_id=payload.request_id,
        session_id=session_id,
        decision=payload.decision.value,
    )
    if not approval["created"]:
        if approval["decision"] != payload.decision.value:
            return _error_response(
                status_code=status.HTTP_409_CONFLICT,
                code="approval_decision_conflict",
                message="A different approval decision was already submitted.",
                session_id=session_id,
                request_id=payload.request_id,
            )
        current = runtime.store.get_conversation_run(payload.request_id)
        return ApprovalSubmitResponse(
            session_id=session_id,
            request_id=payload.request_id,
            decision=payload.decision,
            status=current["status"],
        )

    runtime.store.update_conversation_run_status(
        payload.request_id,
        RunStatus.RUNNING.value,
    )
    runtime.store.append_conversation_event(
        request_id=payload.request_id,
        session_id=session_id,
        event=StreamEventType.TRACE.value,
        data={
            "phase": "approval_submitted",
            "decision": payload.decision.value,
            "status": RunStatus.RUNNING.value,
        },
    )
    background_tasks.add_task(
        execute_approval,
        runtime=runtime,
        request_id=payload.request_id,
        session_id=session_id,
        thread_id=conversation["thread_id"],
        decision=payload.decision.value,
    )
    return ApprovalSubmitResponse(
        session_id=session_id,
        request_id=payload.request_id,
        decision=payload.decision,
        status=RunStatus.RUNNING,
    )
