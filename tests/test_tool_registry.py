import pytest
from pydantic import BaseModel

from app.schemas import RiskLevel
from app.tools import (
    DuplicateToolError,
    ToolCategory,
    ToolDefinition,
    ToolErrorCode,
    ToolPermission,
    ToolRegistry,
    ToolResult,
)


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


def echo_handler(input_data: BaseModel) -> ToolResult:
    data = EchoInput.model_validate(input_data)
    return ToolResult.ok(
        data={"text": data.text},
        risk_level=RiskLevel.L0_READ_ONLY,
    )


def broken_handler(input_data: BaseModel) -> ToolResult:
    raise RuntimeError("database unavailable")


def make_echo_tool(
    *,
    name: str = "echo",
    permission: ToolPermission = ToolPermission.AUTO_EXECUTE,
    risk_level: RiskLevel = RiskLevel.L0_READ_ONLY,
):
    return ToolDefinition(
        name=name,
        description="Echo validated input text.",
        category=ToolCategory.SYSTEM,
        input_model=EchoInput,
        output_model=EchoOutput,
        risk_level=risk_level,
        permission=permission,
        handler=echo_handler,
    )


def test_registry_registers_and_lists_tools() -> None:
    registry = ToolRegistry()
    tool = make_echo_tool()

    registry.register(tool)

    assert registry.get("echo") == tool
    assert registry.list_tools() == [tool]


def test_registry_lists_only_allowed_tools() -> None:
    registry = ToolRegistry()
    registry.register(make_echo_tool(name="allowed"))
    registry.register(make_echo_tool(name="blocked"))

    tools = registry.list_tools(allowed_tool_names={"allowed"})

    assert [tool.name for tool in tools] == ["allowed"]


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    registry.register(make_echo_tool())

    with pytest.raises(DuplicateToolError):
        registry.register(make_echo_tool())


def test_registry_returns_structured_error_for_missing_tool() -> None:
    registry = ToolRegistry()

    result = registry.execute("missing_tool", {"text": "hello"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TOOL_NOT_FOUND
    assert result.error.recoverable is False
    assert result.risk_level is RiskLevel.L3_FORBIDDEN


def test_registry_blocks_tool_outside_allowlist() -> None:
    registry = ToolRegistry()
    registry.register(make_echo_tool())

    result = registry.execute(
        "echo",
        {"text": "hello"},
        allowed_tool_names={"other_tool"},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert result.error.recoverable is False
    assert result.error.details == {"tool_name": "echo"}
    assert result.risk_level is RiskLevel.L3_FORBIDDEN


def test_registry_validates_arguments_before_execution() -> None:
    registry = ToolRegistry()
    registry.register(make_echo_tool())

    result = registry.execute("echo", {"wrong": "hello"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    assert result.error.details["errors"][0]["loc"] == ("text",)


def test_registry_executes_auto_approved_tool() -> None:
    registry = ToolRegistry()
    registry.register(make_echo_tool())

    result = registry.execute("echo", {"text": "hello"})

    assert result.success is True
    assert result.data == {"text": "hello"}
    assert result.trace is not None
    assert result.trace.tool_name == "echo"


def test_registry_blocks_approval_required_tool_without_approval() -> None:
    registry = ToolRegistry()
    registry.register(
        make_echo_tool(
            permission=ToolPermission.REQUIRE_APPROVAL,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        )
    )

    result = registry.execute("echo", {"text": "hello"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert result.risk_level is RiskLevel.L2_CONTROLLED_WRITE


def test_registry_executes_approval_required_tool_when_approved() -> None:
    registry = ToolRegistry()
    registry.register(
        make_echo_tool(
            permission=ToolPermission.REQUIRE_APPROVAL,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        )
    )

    result = registry.execute("echo", {"text": "hello"}, approved=True)

    assert result.success is True
    assert result.data == {"text": "hello"}


def test_registry_blocks_forbidden_tool() -> None:
    registry = ToolRegistry()
    registry.register(
        make_echo_tool(
            permission=ToolPermission.FORBIDDEN,
            risk_level=RiskLevel.L3_FORBIDDEN,
        )
    )

    result = registry.execute("echo", {"text": "hello"}, approved=True)

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert result.error.recoverable is False


def test_registry_wraps_handler_exceptions() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="broken",
            description="A tool that raises.",
            category=ToolCategory.SYSTEM,
            input_model=EchoInput,
            output_model=EchoOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            handler=broken_handler,
        )
    )

    result = registry.execute("broken", {"text": "hello"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_ERROR
    assert result.error.details == {"error": "database unavailable"}


def test_registry_rejects_invalid_success_output() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="invalid_output",
            description="Return the wrong success shape.",
            category=ToolCategory.SYSTEM,
            input_model=EchoInput,
            output_model=EchoOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            handler=lambda _: ToolResult.ok(
                data={"wrong": "shape"},
                risk_level=RiskLevel.L0_READ_ONLY,
            ),
        )
    )

    result = registry.execute("invalid_output", {"text": "hello"})

    assert result.success is False
    assert result.error.code is ToolErrorCode.EXECUTION_ERROR
    assert result.error.details["errors"][0]["location"] == ["text"]
