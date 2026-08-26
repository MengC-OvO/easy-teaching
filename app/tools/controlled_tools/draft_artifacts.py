"""Scoped, read-only access to generated conversation drafts."""

import inspect
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolErrorCode,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


class ReadDraftArtifactInput(BaseModel):
    source_request_id: str = Field(
        min_length=1,
        max_length=128,
        description="Exact source_request_id from conversation workspace metadata.",
    )


class ReadDraftArtifactOutput(BaseModel):
    source_request_id: str
    title: Optional[str] = None
    content: str
    content_chars: int = Field(ge=1)
    created_at: str
    status: str


def build_read_draft_artifact_tool(store: Any) -> ToolDefinition:
    async def async_runtime_handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not all(
            (context.teacher_id, context.class_id, context.session_id)
        ):
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Draft access requires a trusted conversation scope.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
            )

        data = ReadDraftArtifactInput.model_validate(input_data)
        reader = getattr(store, "get_conversation_artifact", None)
        if reader is None:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="Draft artifact storage is unavailable.",
                risk_level=RiskLevel.L0_READ_ONLY,
                recoverable=True,
            )
        artifact = reader(
            source_request_id=data.source_request_id,
            session_id=context.session_id,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
        )
        if inspect.isawaitable(artifact):
            artifact = await artifact
        if artifact is None:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="The requested draft is unavailable in this conversation.",
                risk_level=RiskLevel.L0_READ_ONLY,
                recoverable=True,
            )
        output = ReadDraftArtifactOutput.model_validate(artifact)
        return ToolResult.ok(
            data=output.model_dump(mode="json"),
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="read_draft_artifact",
        description=(
            "Read the complete text of one generated conversation draft by its exact "
            "source_request_id. Use only when revising, expanding, shortening, "
            "comparing, or quoting an existing unsaved draft; do not use for saved "
            "observations or educational records."
        ),
        category=ToolCategory.DRAFT,
        input_model=ReadDraftArtifactInput,
        output_model=ReadDraftArtifactOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        max_identical_calls_per_run=1,
        async_runtime_handler=async_runtime_handler,
    )
