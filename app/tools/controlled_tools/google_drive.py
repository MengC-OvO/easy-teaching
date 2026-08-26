"""Narrow, policy-controlled wrappers around Google Workspace MCP Drive tools."""

import base64
import hashlib
import inspect
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolErrorCode,
    ToolExecutionContext,
    ToolKind,
    ToolPermission,
    ToolResult,
)
from app.tools.mcp_adapter import MCPClientProtocol


class SearchGoogleDriveInput(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    file_type: Optional[str] = Field(default=None, max_length=80)
    limit: int = Field(default=10, ge=1, le=20)


class SearchGoogleDriveOutput(BaseModel):
    results_text: str
    provider: str = "google_drive"


class UploadExportToGoogleDriveInput(BaseModel):
    export_id: str = Field(min_length=1, max_length=128)
    folder_id: str = Field(default="root", min_length=1, max_length=256)
    file_name: Optional[str] = Field(default=None, min_length=1, max_length=255)


class UploadExportToGoogleDriveOutput(BaseModel):
    result_text: str
    provider: str = "google_drive"


def build_google_drive_tools(
    store: Any,
    *,
    client: MCPClientProtocol,
    user_google_email: str,
    timeout_seconds: float = 45.0,
    export_root: Path | None = None,
) -> List[ToolDefinition]:
    root = (export_root or Path("data/exports")).resolve()

    async def search_handler(input_data: BaseModel) -> ToolResult:
        data = SearchGoogleDriveInput.model_validate(input_data)
        arguments: Dict[str, Any] = {
            "user_google_email": user_google_email,
            "query": data.query,
            "page_size": data.limit,
            "detailed": True,
            "include_trashed": False,
        }
        if data.file_type:
            arguments["file_type"] = data.file_type
        result = await client.call_tool(
            server_name="google_workspace",
            tool_name="search_drive_files",
            arguments=arguments,
        )
        return ToolResult.ok(
            data={"results_text": result.get("text", ""), "provider": "google_drive"},
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    async def upload_handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.teacher_id or not context.class_id:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Google Drive upload requires a trusted teacher and class scope.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
            )
        data = UploadExportToGoogleDriveInput.model_validate(input_data)
        record = store.get_record_export(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            export_id=data.export_id,
        )
        if inspect.isawaitable(record):
            record = await record
        path = Path(record["storage_path"]).resolve()
        if root not in path.parents or not path.is_file():
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Export file is outside the managed export directory.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
            )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record["checksum"]:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message="Export checksum changed; create a fresh export before uploading.",
                risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                recoverable=True,
            )
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        result = await client.call_tool(
            server_name="google_workspace",
            tool_name="create_drive_file",
            arguments={
                "user_google_email": user_google_email,
                "file_name": data.file_name or path.name,
                "folder_id": data.folder_id,
                "mime_type": mime_type,
                "base64_content": base64.b64encode(raw).decode("ascii"),
                "content_mime_type": mime_type,
            },
        )
        return ToolResult.ok(
            data={"result_text": result.get("text", ""), "provider": "google_drive"},
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        )

    return [
        ToolDefinition(
            name="search_google_drive",
            description=(
                "Search the teacher's authorised Google Drive for files. Use only when "
                "the teacher explicitly asks to find or view something in Google Drive."
            ),
            category=ToolCategory.SYSTEM,
            input_model=SearchGoogleDriveInput,
            output_model=SearchGoogleDriveOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            kind=ToolKind.MCP,
            domain=ToolDomain.EXTERNAL,
            parallel_safe=True,
            timeout_seconds=timeout_seconds,
            async_handler=search_handler,
        ),
        ToolDefinition(
            name="upload_export_to_google_drive",
            description=(
                "Upload an already-created, teacher-authorised record export to Google "
                "Drive. Accept only an export_id; never arbitrary local file paths."
            ),
            category=ToolCategory.DRAFT,
            input_model=UploadExportToGoogleDriveInput,
            output_model=UploadExportToGoogleDriveOutput,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
            permission=ToolPermission.REQUIRE_APPROVAL,
            completion_aliases=(
                "上传到google drive",
                "保存到google drive",
                "upload to google drive",
                "save to google drive",
            ),
            kind=ToolKind.MCP,
            domain=ToolDomain.EXTERNAL,
            parallel_safe=False,
            timeout_seconds=timeout_seconds,
            async_runtime_handler=upload_handler,
        ),
    ]
