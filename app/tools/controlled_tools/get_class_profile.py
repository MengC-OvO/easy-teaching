from typing import List

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.services import EduFlowStore
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolErrorCode,
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


def build_get_class_profile_tool(store: EduFlowStore) -> ToolDefinition:
    def handler(input_data: BaseModel) -> ToolResult:
        data = GetClassProfileInput.model_validate(input_data)
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

    return ToolDefinition(
        name="get_class_profile",
        description="Read a synthetic class profile by class id.",
        category=ToolCategory.CLASS_PROFILE,
        input_model=GetClassProfileInput,
        output_model=GetClassProfileOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )
