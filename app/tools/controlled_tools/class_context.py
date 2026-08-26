import inspect
from typing import Any, Dict, List, Optional

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


class GetClassContextInput(BaseModel):
    memory_query: Optional[str] = Field(default=None, max_length=300)
    memory_limit: int = Field(default=4, ge=0, le=8)
    include_children: bool = False


class GetClassContextOutput(BaseModel):
    centre_id: str
    class_id: str
    name: str
    age_group: str
    child_count: int = Field(ge=0)
    current_focus: List[str] = Field(default_factory=list)
    children: List[Dict[str, str]] = Field(default_factory=list)
    class_memories: List[Dict[str, Any]] = Field(default_factory=list)


def build_get_class_context_tool(store: Any) -> ToolDefinition:
    async def async_runtime_handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        data = GetClassContextInput.model_validate(input_data)
        if not context.teacher_id or not context.class_id:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Class context requires a trusted teacher and class scope.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
            )
        result = store.get_class_context(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            memory_query=data.memory_query,
            memory_limit=data.memory_limit,
            include_children=data.include_children,
        )
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="The trusted class context is unavailable.",
                risk_level=RiskLevel.L0_READ_ONLY,
            )
        return ToolResult.ok(data=result, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="get_class_context",
        description=(
            "Read the authorised class age group, child count, teaching focus and "
            "durable class facts. Use include_children only to link a record to a "
            "pseudonymous child ID. Never use it for event time/details or observations."
        ),
        category=ToolCategory.CLASS_PROFILE,
        input_model=GetClassContextInput,
        output_model=GetClassContextOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        async_runtime_handler=async_runtime_handler,
    )
