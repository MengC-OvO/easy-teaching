"""把已批准的 MCP 能力适配为普通 ToolDefinition。"""

from typing import Any, Dict, Protocol, Type

from pydantic import BaseModel

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolKind,
    ToolPermission,
    ToolResult,
)


class MCPClientProtocol(Protocol):
    async def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...


def build_read_only_mcp_tool(
    *,
    name: str,
    description: str,
    server_name: str,
    remote_tool_name: str,
    input_model: Type[BaseModel],
    output_model: Type[BaseModel],
    client: MCPClientProtocol,
    domain: ToolDomain = ToolDomain.EXTERNAL,
    timeout_seconds: float = 20.0,
) -> ToolDefinition:
    """MCP 只改变传输方式，不绕过 Registry 权限和 Pydantic 校验。"""

    async def handler(arguments: BaseModel) -> ToolResult:
        data = await client.call_tool(
            server_name=server_name,
            tool_name=remote_tool_name,
            arguments=arguments.model_dump(mode="json"),
        )
        return ToolResult.ok(data=data, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name=name,
        description=description,
        category=ToolCategory.SYSTEM,
        input_model=input_model,
        output_model=output_model,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        kind=ToolKind.MCP,
        domain=domain,
        parallel_safe=True,
        timeout_seconds=timeout_seconds,
        async_handler=handler,
    )
