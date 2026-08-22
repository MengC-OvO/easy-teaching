"""Accept teacher messages for execution on an EasyTeaching session thread."""

from typing import Optional, Union
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, Request, status
from fastapi.responses import JSONResponse

from app.api.dependencies import get_runtime
from app.api.errors import ConversationSessionBusyError
from app.api.auth import CurrentUser, get_current_user, require_session_owner
from app.api.execution import execute_message
from app.integrations.input_safety import InputSafetyRejected, prepare_user_input
from app.integrations.privacy_gateway_client import PrivacyGatewayUnavailableError
from app.schemas import (
    ApiErrorDetail,
    ApiErrorResponse,
    MessageAcceptedResponse,
    MessageCreateRequest,
    StreamEventType,
)


router = APIRouter(prefix="/sessions", tags=["messages"])


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict,
    request_id: Optional[str] = None,
    recoverable: bool = False,
) -> JSONResponse:
    error = ApiErrorResponse(
        request_id=request_id,
        error=ApiErrorDetail(
            code=code,
            message=message,
            recoverable=recoverable,
            details=details,
        ),
    )
    return JSONResponse(
        status_code=status_code,
        content=error.model_dump(mode="json"),
    )


@router.post(
    "/{session_id}/messages",
    response_model=MessageAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ApiErrorResponse},
        status.HTTP_409_CONFLICT: {"model": ApiErrorResponse},
    },
)
async def create_message(
    session_id: str,
    payload: MessageCreateRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> Union[MessageAcceptedResponse, JSONResponse]:
    runtime = get_runtime(request)
    conversation = await runtime.store.get_conversation_session(session_id)
    if conversation is None:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="session_not_found",
            message="The requested session does not exist.",
            details={"session_id": session_id},
            request_id=payload.request_id,
        )
    require_session_owner(conversation, current_user)

    request_id = payload.request_id or str(uuid4())
    existing = await runtime.store.get_conversation_run(request_id)
    if existing is not None:
        if existing["session_id"] != session_id:
            return _error_response(
                status_code=status.HTTP_409_CONFLICT,
                code="request_id_conflict",
                message="The request_id already belongs to another session.",
                details={"session_id": session_id},
                request_id=request_id,
            )
        return MessageAcceptedResponse(
            session_id=session_id,
            request_id=request_id,
            status=existing["status"],
        )

    active = await runtime.store.get_active_conversation_run(session_id)
    if active is not None:
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="session_busy",
            message="The session already has an active run.",
            details={"active_request_id": active["request_id"]},
            request_id=request_id,
        )

    try:
        prepared = await prepare_user_input(
            mode=getattr(runtime, "privacy_gateway_mode", "disabled"),
            client=getattr(runtime, "privacy_gateway_client", None),
            session_id=session_id,
            text=payload.message,
        )
    except InputSafetyRejected as error:
        inspection = error.inspection
        response_status = (
            status.HTTP_403_FORBIDDEN
            if inspection.action.value == "block"
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        return _error_response(
            status_code=response_status,
            code=(
                "safety_blocked"
                if inspection.action.value == "block"
                else "safety_clarification_required"
            ),
            message=(
                "The request was blocked by the local safety boundary."
                if inspection.action.value == "block"
                else "Please clarify or rephrase the request before continuing."
            ),
            details={
                "action": inspection.action.value,
                "reason_code": inspection.reason_code,
                "signals": inspection.signals.model_dump(mode="json"),
            },
            request_id=request_id,
            recoverable=inspection.action.value == "clarify",
        )
    except PrivacyGatewayUnavailableError:
        return _error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="privacy_gateway_unavailable",
            message="The local privacy gateway could not safely process the request.",
            details={"mode": getattr(runtime, "privacy_gateway_mode", "disabled")},
            request_id=request_id,
            recoverable=True,
        )

    try:
        run = await runtime.store.create_conversation_run(
            request_id=request_id,
            session_id=session_id,
        )
    except ConversationSessionBusyError as error:
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="session_busy",
            message="The session already has an active run.",
            details={"active_request_id": error.active_request_id},
            request_id=request_id,
        )

    if not run["created"]:
        if run["session_id"] != session_id:
            return _error_response(
                status_code=status.HTTP_409_CONFLICT,
                code="request_id_conflict",
                message="The request_id already belongs to another session.",
                details={"session_id": session_id},
                request_id=request_id,
            )
        return MessageAcceptedResponse(
            session_id=session_id,
            request_id=request_id,
            status=run["status"],
        )

    await runtime.store.append_conversation_event(
        request_id=request_id,
        session_id=session_id,
        event=StreamEventType.RUN_STARTED.value,
        data={"status": run["status"]},
    )
    background_tasks.add_task(
        execute_message,
        runtime=runtime,
        request_id=request_id,
        session_id=session_id,
        thread_id=conversation["thread_id"],
        teacher_id=conversation["teacher_id"],
        class_id=conversation["class_id"],
        message=prepared.forwarded_text,
        privacy_mapping_id=prepared.mapping_id,
    )
    return MessageAcceptedResponse(
        session_id=session_id,
        request_id=request_id,
        status=run["status"],
    )
