from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.services import EduFlowStore
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolPermission,
    ToolResult,
)


class SaveDraftInput(BaseModel):
    draft_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    draft_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SaveDraftOutput(BaseModel):
    draft_id: str
    draft_type: str
    title: str
    status: str


def build_save_draft_tool(store: EduFlowStore) -> ToolDefinition:
    def handler(input_data: BaseModel) -> ToolResult:
        data = SaveDraftInput.model_validate(input_data)
        saved = store.save_draft(
            draft_id=data.draft_id,
            draft_type=data.draft_type,
            title=data.title,
            content=data.content,
            idempotency_key=data.idempotency_key,
        )
        return ToolResult.ok(data=saved, risk_level=RiskLevel.L2_CONTROLLED_WRITE)

    return ToolDefinition(
        name="save_draft",
        description="Save a draft record after teacher approval.",
        category=ToolCategory.DRAFT,
        input_model=SaveDraftInput,
        output_model=SaveDraftOutput,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        handler=handler,
    )
