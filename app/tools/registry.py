import asyncio
import inspect
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
                message=f"Tool is not allowed for this Agent execution: {name}",
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

        permission = tool.permission_for(raw_args)
        risk_level = tool.risk_for(raw_args)
        trace = ToolTrace(
            tool_name=tool.name,
            risk_level=risk_level,
            permission=permission,
            message="Tool execution requested.",
        )

        if permission is ToolPermission.FORBIDDEN:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Tool is forbidden: {tool.name}",
                risk_level=risk_level,
                recoverable=False,
                details={"tool_name": tool.name},
                trace=trace,
            )

        if permission is ToolPermission.REQUIRE_APPROVAL and not approved:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Tool requires approval before execution: {tool.name}",
                risk_level=risk_level,
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
                risk_level=risk_level,
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
                risk_level=risk_level,
                recoverable=True,
                details={"error": str(error)},
                trace=trace,
            )

        if result.success:
            try:
                validated_output = tool.output_model.model_validate(result.data)
            except ValidationError as error:
                return ToolResult.fail(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"Tool returned an invalid output: {tool.name}",
                    risk_level=risk_level,
                    recoverable=True,
                    details={
                        "errors": [
                            {
                                "location": list(item["loc"]),
                                "type": item["type"],
                                "message": item["msg"],
                            }
                            for item in error.errors()
                        ]
                    },
                    trace=trace,
                )
            result = result.model_copy(
                update={"data": validated_output.model_dump(mode="json")}
            )

        if result.trace is None:
            result.trace = trace
        return result

    async def execute_async(
        self,
        name: str,
        raw_args: Dict[str, Any],
        *,
        approved: bool = False,
        allowed_tool_names: Optional[Iterable[str]] = None,
        execution_context: Optional[ToolExecutionContext] = None,
    ) -> ToolResult:
        tool = self.get(name)
        timeout_seconds = tool.timeout_seconds if tool is not None else 30.0
        try:
            return await asyncio.wait_for(
                self._execute_native_async(
                    name,
                    raw_args,
                    approved=approved,
                    allowed_tool_names=allowed_tool_names,
                    execution_context=execution_context,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ToolResult.fail(
                code=ToolErrorCode.TIMEOUT,
                message=f"Tool execution timed out: {name}",
                risk_level=(
                    tool.risk_for(raw_args)
                    if tool is not None
                    else RiskLevel.L3_FORBIDDEN
                ),
                recoverable=True,
                details={"tool_name": name, "timeout_seconds": timeout_seconds},
            )

    async def _execute_native_async(
        self,
        name: str,
        raw_args: Dict[str, Any],
        *,
        approved: bool,
        allowed_tool_names: Optional[Iterable[str]],
        execution_context: Optional[ToolExecutionContext],
    ) -> ToolResult:
        if not self._is_allowed(name, allowed_tool_names):
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Tool is not allowed for this Agent execution: {name}",
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
        permission = tool.permission_for(raw_args)
        risk_level = tool.risk_for(raw_args)
        trace = ToolTrace(
            tool_name=tool.name,
            risk_level=risk_level,
            permission=permission,
            message="Tool execution requested.",
        )
        if permission is ToolPermission.FORBIDDEN or (
            permission is ToolPermission.REQUIRE_APPROVAL and not approved
        ):
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=f"Tool is not permitted: {tool.name}",
                risk_level=risk_level,
                recoverable=permission is ToolPermission.REQUIRE_APPROVAL,
                details={"tool_name": tool.name},
                trace=trace,
            )
        try:
            validated_args = tool.input_model.model_validate(raw_args)
        except ValidationError as error:
            return ToolResult.fail(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=f"Invalid arguments for tool: {tool.name}",
                risk_level=risk_level,
                details={"errors": error.errors()},
                trace=trace,
            )
        try:
            if tool.async_runtime_handler is not None:
                result = await tool.async_runtime_handler(
                    validated_args,
                    execution_context or ToolExecutionContext(),
                )
            elif tool.async_handler is not None:
                result = await tool.async_handler(validated_args)
            elif tool.runtime_handler is not None:
                result = tool.runtime_handler(
                    validated_args,
                    execution_context or ToolExecutionContext(),
                )
            else:
                assert tool.handler is not None
                result = tool.handler(validated_args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as error:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Tool execution failed: {tool.name}",
                risk_level=risk_level,
                details={"error": str(error)},
                trace=trace,
            )
        if result.success:
            try:
                output = tool.output_model.model_validate(result.data)
            except ValidationError as error:
                return ToolResult.fail(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"Tool returned an invalid output: {tool.name}",
                    risk_level=risk_level,
                    details={"errors": error.errors()},
                    trace=trace,
                )
            result = result.model_copy(update={"data": output.model_dump(mode="json")})
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
