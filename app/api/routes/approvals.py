"""Approve or reject a frozen controlled-write action."""

import hashlib
import json

from typing import Optional, Union

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from app.api.auth import CurrentUser, get_current_user, require_session_owner
from app.api.dependencies import get_runtime
from app.schemas import (
    ApiErrorDetail,
    ApiErrorResponse,
    ApprovalDecision,
    ApprovalStatus,
    ApprovalSubmitRequest,
    ApprovalSubmitResponse,
    RunStatus,
    StreamEventType,
)
from app.tools import ToolExecutionContext


router = APIRouter(prefix="/sessions", tags=["approvals"])


def _error(code: str, message: str, request_id: str, *, status_code: int = 409):
    return JSONResponse(
        status_code=status_code,
        content=ApiErrorResponse(
            request_id=request_id,
            error=ApiErrorDetail(code=code, message=message, recoverable=True),
        ).model_dump(mode="json"),
    )


@router.post(
    "/{session_id}/approvals",
    response_model=ApprovalSubmitResponse,
    responses={409: {"model": ApiErrorResponse}, 404: {"model": ApiErrorResponse}},
)
async def submit_approval(
    session_id: str,
    payload: ApprovalSubmitRequest,
    request: Request,
    current_user: Optional[CurrentUser] = Depends(get_current_user),
) -> Union[ApprovalSubmitResponse, JSONResponse]:
    runtime = get_runtime(request)
    conversation = await runtime.store.get_conversation_session(session_id)
    if conversation is None:
        return _error("session_not_found", "The session does not exist.", payload.request_id, status_code=404)
    require_session_owner(conversation, current_user)
    run = await runtime.store.get_conversation_run(payload.request_id)
    result = await runtime.store.get_conversation_run_result(payload.request_id)
    if run is None or result is None or run["session_id"] != session_id:
        return _error("request_not_found", "The request does not exist in this session.", payload.request_id, status_code=404)
    if run["status"] != RunStatus.WAITING_FOR_APPROVAL.value:
        return _error("approval_conflict", "The request is not waiting for approval.", payload.request_id)
    approval = result["approval"]
    action_id = approval.get("action_id")
    if not action_id:
        return _error("approval_missing_action", "The approval has no frozen action.", payload.request_id)
    action = await runtime.store.get_tool_action_request(action_id)
    if action is None or action["request_id"] != payload.request_id or action["session_id"] != session_id:
        return _error("approval_action_mismatch", "The frozen action does not match this request.", payload.request_id)

    if payload.decision is ApprovalDecision.REJECT:
        try:
            await runtime.store.finish_tool_action_request(action_id, status="rejected")
        except ValueError as error:
            return _error("approval_conflict", str(error), payload.request_id)
        approval = {**approval, "status": ApprovalStatus.REJECTED.value}
    else:
        try:
            action = await runtime.store.claim_tool_action_request(action_id)
        except ValueError as error:
            return _error("approval_conflict", str(error), payload.request_id)
        serialized = json.dumps(
            action["arguments"], ensure_ascii=False, sort_keys=True, default=str
        )
        if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != action["arguments_hash"]:
            await runtime.store.finish_tool_action_request(action_id, status="failed")
            await runtime.store.save_conversation_run_result(
                request_id=payload.request_id,
                session_id=session_id,
                draft=result["draft"],
                approval={**approval, "status": ApprovalStatus.FAILED.value},
                citations=result["citations"],
            )
            await runtime.store.update_conversation_run_status(
                payload.request_id, RunStatus.FAILED.value
            )
            await runtime.store.append_conversation_event(
                request_id=payload.request_id,
                session_id=session_id,
                event=StreamEventType.FAILED.value,
                data={"status": RunStatus.FAILED.value, "tool_name": action["tool_name"]},
            )
            return _error(
                "approval_action_corrupt",
                "The frozen action failed its integrity check.",
                payload.request_id,
            )
        tool_result = await runtime.tool_registry.execute_async(
            action["tool_name"],
            action["arguments"],
            approved=True,
            execution_context=ToolExecutionContext(
                teacher_id=action["teacher_id"],
                class_id=action["class_id"],
                session_id=action["session_id"],
                request_id=action["request_id"],
            ),
        )
        if not tool_result.success:
            await runtime.store.finish_tool_action_request(
                action_id,
                status="failed",
                result=tool_result.model_dump(mode="json"),
            )
            await runtime.store.update_conversation_run_status(
                payload.request_id, RunStatus.FAILED.value
            )
            await runtime.store.save_conversation_run_result(
                request_id=payload.request_id,
                session_id=session_id,
                draft=result["draft"],
                approval={
                    **approval,
                    "status": ApprovalStatus.FAILED.value,
                    "result": {
                        "error": (
                            tool_result.error.model_dump(mode="json")
                            if tool_result.error
                            else {}
                        )
                    },
                },
                citations=result["citations"],
            )
            await runtime.store.append_conversation_event(
                request_id=payload.request_id,
                session_id=session_id,
                event=StreamEventType.FAILED.value,
                data={"status": RunStatus.FAILED.value, "tool_name": action["tool_name"]},
            )
            return _error(
                "approved_tool_failed",
                tool_result.error.message if tool_result.error else "The approved action failed.",
                payload.request_id,
            )
        await runtime.store.finish_tool_action_request(
            action_id,
            status="executed",
            result=tool_result.data,
        )
        approval = {
            **approval,
            "status": ApprovalStatus.APPROVED.value,
            "result": tool_result.data,
        }

    await runtime.store.save_conversation_run_result(
        request_id=payload.request_id,
        session_id=session_id,
        draft=result["draft"],
        approval=approval,
        citations=result["citations"],
    )
    await runtime.store.update_conversation_run_status(
        payload.request_id, RunStatus.COMPLETED.value
    )
    await runtime.store.append_conversation_event(
        request_id=payload.request_id,
        session_id=session_id,
        event=StreamEventType.COMPLETED.value,
        data={
            "status": RunStatus.COMPLETED.value,
            "approval_decision": payload.decision.value,
            "tool_name": action["tool_name"],
        },
    )
    return ApprovalSubmitResponse(
        session_id=session_id,
        request_id=payload.request_id,
        decision=payload.decision,
        status=RunStatus.COMPLETED,
    )
