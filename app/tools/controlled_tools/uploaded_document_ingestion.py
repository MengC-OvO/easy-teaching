"""Approval-controlled ingestion of uploaded centre documents."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.controlled_tools.upload_scope import require_upload_scope
from app.tools.definition import (
    PreparedToolAction,
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


class IngestUploadedDocumentInput(BaseModel):
    file_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str = Field(min_length=2, max_length=300)


class IngestUploadedDocumentOutput(BaseModel):
    source_id: str
    title: str
    chunk_count: int
    source_type: Literal["centre"]
    index_mode: Literal["tenant_local_hybrid"]
    content_hash: str


def build_ingest_uploaded_document_tool(scoped_knowledge: Any) -> ToolDefinition:
    async def prepare(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> PreparedToolAction:
        data = IngestUploadedDocumentInput.model_validate(input_data)
        if require_upload_scope(context):
            raise ValueError("Trusted upload scope is required")
        preview = scoped_knowledge.preview(
            file_id=data.file_id,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            session_id=context.session_id,
        )
        return PreparedToolAction(
            arguments=data.model_dump(mode="json"),
            preview={
                **preview,
                "title": data.title,
                "index_mode": "tenant_local_hybrid",
            },
        )

    async def run(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = require_upload_scope(context)
        if denied:
            return denied
        data = IngestUploadedDocumentInput.model_validate(input_data)
        result = await scoped_knowledge.ingest(
            file_id=data.file_id,
            title=data.title,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            session_id=context.session_id,
        )
        return ToolResult.ok(data=result, risk_level=RiskLevel.L2_CONTROLLED_WRITE)

    return ToolDefinition(
        name="ingest_uploaded_document",
        description=(
            "After teacher approval, add an uploaded centre document to that "
            "teacher/class's isolated local BM25 and Chroma indexes. Reading alone "
            "does not index it."
        ),
        category=ToolCategory.FILE,
        input_model=IngestUploadedDocumentInput,
        output_model=IngestUploadedDocumentOutput,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        completion_aliases=(
            "加入知识库",
            "添加到知识库",
            "add to the knowledge base",
            "index this document",
        ),
        domain=ToolDomain.LOCAL,
        parallel_safe=False,
        async_runtime_handler=run,
        approval_preparation_handler=prepare,
    )
