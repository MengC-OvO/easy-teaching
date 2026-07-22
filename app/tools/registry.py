from typing import Any, Dict, Iterable, List, Optional, Set

from pydantic import ValidationError

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolDefinition,
    ToolExecutionContext,
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

    def list_tools(
        self,
        *,
        allowed_tool_names: Optional[Iterable[str]] = None,
    ) -> List[ToolDefinition]:
        allowed = self._allowed_tool_name_set(allowed_tool_names)
        if allowed is None:
            return list(self._tools.values())
        return [tool for name, tool in self._tools.items() if name in allowed]

    def execute(
        self,
        name: str,
        raw_args: Dict[str, Any],
        *,
        approved: bool = False,
        allowed_tool_names: Optional[Iterable[str]] = None,
        execution_context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        if not self._is_allowed(name, allowed_tool_names):
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Tool is not allowed in this workflow: {name}",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
                details={"tool_name": name},
            )

        tool = self.get(name)
        if tool is None:
            return ToolResult.fail(
                code=ToolErrorCode.TOOL_NOT_FOUND,
                message=f"Tool is not registered: {name}",
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
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
            if tool.runtime_handler is not None:
                result = tool.runtime_handler(
                    validated_args,
                    execution_context or ToolExecutionContext(),
                )
            else:
                assert tool.handler is not None
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

    def _allowed_tool_name_set(
        self,
        allowed_tool_names: Optional[Iterable[str]],
    ) -> Optional[Set[str]]:
        if allowed_tool_names is None:
            return None
        return set(allowed_tool_names)

    def _is_allowed(
        self,
        name: str,
        allowed_tool_names: Optional[Iterable[str]],
    ) -> bool:
        allowed = self._allowed_tool_name_set(allowed_tool_names)
        return allowed is None or name in allowed
