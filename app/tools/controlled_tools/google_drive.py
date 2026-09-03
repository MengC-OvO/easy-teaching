"""One policy-controlled gateway for dynamically discovered Drive MCP tools."""

import base64
import copy
import hashlib
import inspect
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as JsonSchemaError
from pydantic import BaseModel, Field, model_validator

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
from app.tools.mcp_adapter import (
    MCPClientProtocol,
    MCPRiskDecision,
    MCPToolInfo,
    classify_mcp_tool,
)


class DriveOperationInput(BaseModel):
    action: Literal["discover", "execute"]
    intent: Optional[str] = Field(default=None, min_length=1, max_length=500)
    tool_name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    arguments: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "DriveOperationInput":
        if self.action == "discover" and not self.intent:
            raise ValueError("intent is required when action=discover")
        if self.action == "execute" and not self.tool_name:
            raise ValueError("tool_name is required when action=execute")
        return self


class DriveOperationOutput(BaseModel):
    action: Literal["discover", "execute"]
    provider: str = "google_drive"
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    selected_tool: Optional[str] = None
    result_text: str = ""
    structured_content: Any = None


class UploadExportArguments(BaseModel):
    """Safe model-facing overlay for MCP create_drive_file."""

    model_config = {"extra": "forbid"}

    export_id: str = Field(min_length=1, max_length=128)
    folder_id: str = Field(default="root", min_length=1, max_length=256)
    file_name: Optional[str] = Field(default=None, min_length=1, max_length=255)


class GoogleDriveMCPGateway:
    """Discover remote tools lazily and execute them through one registered Tool."""

    server_name = "google_workspace"

    def __init__(
        self,
        store: Any,
        *,
        client: MCPClientProtocol,
        user_google_email: str,
        export_root: Path | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.user_google_email = user_google_email
        self.export_root = (export_root or Path("data/exports")).resolve()
        self._catalog: Dict[str, MCPToolInfo] = {}

    async def discover(self) -> List[Dict[str, Any]]:
        tools = await self.client.list_tools(server_name=self.server_name)
        self._catalog = {tool.name: tool for tool in tools}
        return [self._model_tool(tool) for tool in tools]

    def permission_for(self, raw_arguments: Dict[str, Any]) -> ToolPermission:
        return self._decision_for(raw_arguments).permission

    def risk_for(self, raw_arguments: Dict[str, Any]) -> RiskLevel:
        return self._decision_for(raw_arguments).risk_level

    def _decision_for(self, raw_arguments: Dict[str, Any]) -> MCPRiskDecision:
        if raw_arguments.get("action") == "discover":
            return MCPRiskDecision(
                risk_level=RiskLevel.L0_READ_ONLY,
                permission=ToolPermission.AUTO_EXECUTE,
                reason="Discovering the MCP catalog does not change Drive data.",
            )
        tool_name = str(raw_arguments.get("tool_name") or "")
        tool = self._catalog.get(tool_name)
        if tool is None:
            return MCPRiskDecision(
                risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                permission=ToolPermission.REQUIRE_APPROVAL,
                reason="Unknown dynamic calls fail closed until rediscovered.",
            )
        lowered = tool_name.casefold()
        arguments = raw_arguments.get("arguments")
        if any(word in lowered for word in ("delete", "trash", "purge", "permission")):
            return MCPRiskDecision(
                risk_level=RiskLevel.L3_FORBIDDEN,
                permission=ToolPermission.FORBIDDEN,
                reason="Local policy forbids destructive or permission-changing Drive tools.",
            )
        if self._requests_public_access(arguments):
            return MCPRiskDecision(
                risk_level=RiskLevel.L3_FORBIDDEN,
                permission=ToolPermission.FORBIDDEN,
                reason="Local policy forbids public-link or anyone access.",
            )
        return classify_mcp_tool(tool, trusted_server=True)

    @staticmethod
    def _requests_public_access(arguments: Any) -> bool:
        if isinstance(arguments, dict):
            for key, value in arguments.items():
                if str(key).casefold() in {"permission", "role", "visibility", "type"}:
                    if str(value).casefold() in {"anyone", "public", "anyone_with_link"}:
                        return True
                if GoogleDriveMCPGateway._requests_public_access(value):
                    return True
        if isinstance(arguments, list):
            return any(
                GoogleDriveMCPGateway._requests_public_access(item)
                for item in arguments
            )
        return False

    def _model_tool(self, tool: MCPToolInfo) -> Dict[str, Any]:
        decision = self._decision_for(
            {"action": "execute", "tool_name": tool.name, "arguments": {}}
        )
        description = tool.description
        if tool.name == "create_drive_file":
            description = (
                "Upload one EasyTeaching-managed export to Google Drive. "
                "Pass export_id, not file bytes or a local path."
            )
        return {
            "name": tool.name,
            "description": description[:600],
            "input_schema": self._safe_schema(tool),
            "risk_level": decision.risk_level.value,
            "permission": decision.permission.value,
        }

    def _safe_schema(self, tool: MCPToolInfo) -> Dict[str, Any]:
        if tool.name == "create_drive_file":
            return UploadExportArguments.model_json_schema()
        schema = copy.deepcopy(tool.input_schema)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("user_google_email", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [
                name for name in required if name != "user_google_email"
            ]
        return schema

    async def execute(
        self,
        data: DriveOperationInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if data.action == "discover":
            tools = await self.discover()
            return ToolResult.ok(
                data={
                    "action": "discover",
                    "provider": "google_drive",
                    "tools": tools,
                    "result_text": f"Discovered {len(tools)} Google Drive MCP tools.",
                },
                risk_level=RiskLevel.L0_READ_ONLY,
            )

        remote_tool = self._catalog.get(data.tool_name or "")
        if remote_tool is None:
            return ToolResult.fail(
                code=ToolErrorCode.TOOL_NOT_FOUND,
                message=(
                    "Drive MCP tool was not present in the preceding discovery: "
                    f"{data.tool_name}. Discover the catalog before executing."
                ),
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=True,
            )
        decision = self._decision_for(data.model_dump(mode="json"))
        if decision.permission is ToolPermission.FORBIDDEN:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=decision.reason,
                risk_level=decision.risk_level,
                recoverable=False,
            )
        schema = self._safe_schema(remote_tool)
        try:
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(data.arguments)
        except (SchemaError, JsonSchemaError) as error:
            return ToolResult.fail(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=f"Invalid arguments for Drive MCP tool: {data.tool_name}",
                risk_level=decision.risk_level,
                details={"error": error.message},
            )

        if remote_tool.name == "create_drive_file":
            return await self._upload_managed_export(data.arguments, context)
        if self._contains_local_file_payload(data.arguments):
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Raw local paths and file payloads are not accepted by Drive MCP.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
            )
        arguments = dict(data.arguments)
        arguments["user_google_email"] = self.user_google_email
        result = await self.client.call_tool(
            server_name=self.server_name,
            tool_name=remote_tool.name,
            arguments=arguments,
        )
        return ToolResult.ok(
            data={
                "action": "execute",
                "provider": "google_drive",
                "selected_tool": remote_tool.name,
                "result_text": result.get("text", ""),
                "structured_content": result.get("structured_content"),
            },
            risk_level=decision.risk_level,
        )

    @staticmethod
    def _contains_local_file_payload(arguments: Any) -> bool:
        blocked = {
            "file_path",
            "filepath",
            "path",
            "base64_content",
            "fileurl",
            "file_url",
        }
        if isinstance(arguments, dict):
            return any(
                str(key).casefold() in blocked
                or GoogleDriveMCPGateway._contains_local_file_payload(value)
                for key, value in arguments.items()
            )
        if isinstance(arguments, list):
            return any(
                GoogleDriveMCPGateway._contains_local_file_payload(item)
                for item in arguments
            )
        return False

    async def _upload_managed_export(
        self,
        raw_arguments: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not context.teacher_id or not context.class_id:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message="Drive upload requires a trusted teacher and class scope.",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
            )
        data = UploadExportArguments.model_validate(raw_arguments)
        record = self.store.get_record_export(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            export_id=data.export_id,
        )
        if inspect.isawaitable(record):
            record = await record
        path = Path(record["storage_path"]).resolve()
        if self.export_root not in path.parents or not path.is_file():
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
        result = await self.client.call_tool(
            server_name=self.server_name,
            tool_name="create_drive_file",
            arguments={
                "user_google_email": self.user_google_email,
                "file_name": data.file_name or path.name,
                "folder_id": data.folder_id,
                "mime_type": mime_type,
                "base64_content": base64.b64encode(raw).decode("ascii"),
                "content_mime_type": mime_type,
            },
        )
        return ToolResult.ok(
            data={
                "action": "execute",
                "provider": "google_drive",
                "selected_tool": "create_drive_file",
                "result_text": result.get("text", ""),
                "structured_content": result.get("structured_content"),
            },
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        )


def build_google_drive_tool(
    store: Any,
    *,
    client: MCPClientProtocol,
    user_google_email: str,
    timeout_seconds: float = 45.0,
    export_root: Path | None = None,
) -> ToolDefinition:
    gateway = GoogleDriveMCPGateway(
        store,
        client=client,
        user_google_email=user_google_email,
        export_root=export_root,
    )

    async def handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        return await gateway.execute(
            DriveOperationInput.model_validate(input_data),
            context,
        )

    return ToolDefinition(
        name="drive_operation",
        description=(
            "Single gateway for Google Drive MCP. First call action=discover with the "
            "teacher's intent. Read the returned tool names and input schemas. On the "
            "next ReAct step call action=execute with the selected MCP tool_name and "
            "arguments. Discovery is read-only; writes are classified at execution "
            "time and require approval; destructive operations are forbidden."
        ),
        category=ToolCategory.SYSTEM,
        input_model=DriveOperationInput,
        output_model=DriveOperationOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        permission_resolver=gateway.permission_for,
        risk_resolver=gateway.risk_for,
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
        async_runtime_handler=handler,
    )
