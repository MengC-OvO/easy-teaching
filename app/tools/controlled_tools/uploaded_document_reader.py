"""Read a document admitted by the scoped upload store."""

import asyncio
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.services.document_reader import UploadedDocumentReader
from app.services.file_assets import LocalUploadedFileStore
from app.tools.controlled_tools.upload_scope import require_upload_scope
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


class ReadUploadedDocumentInput(BaseModel):
    file_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    max_chars: int = Field(default=12_000, ge=500, le=20_000)


class ReadUploadedDocumentOutput(BaseModel):
    file_id: str
    filename: str
    content_type: str
    sections: List[Dict[str, Any]]
    extracted_chars: int
    truncated: bool


def build_read_uploaded_document_tool(
    file_store: LocalUploadedFileStore,
    reader: UploadedDocumentReader,
) -> ToolDefinition:
    async def run(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = require_upload_scope(context)
        if denied:
            return denied
        data = ReadUploadedDocumentInput.model_validate(input_data)
        record = file_store.get_authorized(
            data.file_id,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            session_id=context.session_id,
            required_category="document",
        )
        result = await asyncio.to_thread(reader.read, record, max_chars=data.max_chars)
        return ToolResult.ok(
            data=result.model_dump(mode="json"),
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="read_uploaded_document",
        description=(
            "Read text from a document the teacher uploaded in this conversation. "
            "Use the opaque file_id returned by the upload API; never invent paths."
        ),
        category=ToolCategory.FILE,
        input_model=ReadUploadedDocumentInput,
        output_model=ReadUploadedDocumentOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        async_runtime_handler=run,
    )
