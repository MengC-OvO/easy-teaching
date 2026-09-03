"""Shared trusted-scope guard for uploaded file capabilities."""

from typing import Optional

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolErrorCode,
    ToolExecutionContext,
    ToolResult,
)


def require_upload_scope(
    context: ToolExecutionContext,
    *,
    session: bool = True,
) -> Optional[ToolResult]:
    if context.teacher_id and context.class_id and (context.session_id or not session):
        return None
    return ToolResult.fail(
        code=ToolErrorCode.PERMISSION_DENIED,
        message="This capability requires trusted teacher, class and session scope.",
        risk_level=RiskLevel.L3_FORBIDDEN,
        recoverable=False,
    )
