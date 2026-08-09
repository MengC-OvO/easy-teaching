import inspect
from typing import Any, List

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


class GetClassProfileInput(BaseModel):
    class_id: str = Field(min_length=1)


class GetClassProfileOutput(BaseModel):
    class_id: str
    name: str
    age_group: str
    child_count: int
    interests: List[str]
    safety_notes: List[str]


def build_get_class_profile_tool(store: Any) -> ToolDefinition:
    def runtime_handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        data = GetClassProfileInput.model_validate(input_data)
        if context.class_id is not None and data.class_id != context.class_id:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Class profile is outside the trusted session scope.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
                details={"class_id": data.class_id},
            )
        profile = store.get_class_profile(data.class_id)
        if profile is None:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Class profile not found: {data.class_id}",
                risk_level=RiskLevel.L0_READ_ONLY,
                recoverable=True,
                details={"class_id": data.class_id},
            )
        return ToolResult.ok(data=profile, risk_level=RiskLevel.L0_READ_ONLY)

    async def async_runtime_handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        data = GetClassProfileInput.model_validate(input_data)
        if context.class_id is not None and data.class_id != context.class_id:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Class profile is outside the trusted session scope.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
                details={"class_id": data.class_id},
            )
        profile = store.get_class_profile(data.class_id)
        if inspect.isawaitable(profile):
            profile = await profile
        if profile is None:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Class profile not found: {data.class_id}",
                risk_level=RiskLevel.L0_READ_ONLY,
                details={"class_id": data.class_id},
            )
        return ToolResult.ok(data=profile, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="get_class_profile",
        description="Read a synthetic class profile by class id.",
        category=ToolCategory.CLASS_PROFILE,
        input_model=GetClassProfileInput,
        output_model=GetClassProfileOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        runtime_handler=runtime_handler,
        async_runtime_handler=async_runtime_handler,
    )
