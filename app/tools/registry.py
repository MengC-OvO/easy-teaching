from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolDefinition,
    ToolErrorCode,
    ToolPermission,
    ToolResult,
    ToolTrace,
)


class DuplicateToolError(ValueError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def execute(
        self,
        name: str,
        raw_args: Dict[str, Any],
        *,
        approved: bool = False,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.fail(
                code=ToolErrorCode.TOOL_NOT_FOUND,
                message=f"Tool is not registered: {name}",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=True,
                details={"tool_name": name},
            )

        trace = ToolTrace(
            tool_name=tool.name,
            risk_level=tool.risk_level,
            permission=tool.permission,
            message="Tool execution requested.",
        )

        if tool.permission is ToolPermission.FORBIDDEN:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Tool is forbidden: {tool.name}",
                risk_level=tool.risk_level,
                recoverable=False,
                details={"tool_name": tool.name},
                trace=trace,
            )

        if tool.requires_approval and not approved:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Tool requires approval before execution: {tool.name}",
                risk_level=tool.risk_level,
                recoverable=True,
                details={"tool_name": tool.name},
                trace=trace,
            )

        try:
            validated_args = tool.input_model.model_validate(raw_args)
        except ValidationError as error:
            return ToolResult.fail(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=f"Invalid arguments for tool: {tool.name}",
                risk_level=tool.risk_level,
                recoverable=True,
                details={"errors": error.errors()},
                trace=trace,
            )

        try:
            result = tool.handler(validated_args)
        except Exception as error:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Tool execution failed: {tool.name}",
                risk_level=tool.risk_level,
                recoverable=True,
                details={"error": str(error)},
                trace=trace,
            )

        if result.trace is None:
            result.trace = trace
        return result
