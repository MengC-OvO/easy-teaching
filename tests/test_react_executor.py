from pydantic import BaseModel

from app.agents import ReActToolExecutor
from app.schemas import (
    ReActAction,
    ReActDecision,
    ReActState,
    StopReason,
    ToolCall,
)
from app.schemas import RiskLevel
from app.tools import ToolCategory, ToolDefinition, ToolPermission, ToolRegistry, ToolResult


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


def echo_handler(input_data: BaseModel) -> ToolResult:
    data = EchoInput.model_validate(input_data)
    return ToolResult.ok(data={"text": data.text}, risk_level=RiskLevel.L0_READ_ONLY)


def make_registry(
    *,
    permission: ToolPermission = ToolPermission.AUTO_EXECUTE,
    risk_level: RiskLevel = RiskLevel.L0_READ_ONLY,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo text.",
            category=ToolCategory.SYSTEM,
            input_model=EchoInput,
            output_model=EchoOutput,
            risk_level=risk_level,
            permission=permission,
            handler=echo_handler,
        )
    )
    return registry


def test_react_tool_executor_runs_tool_call_and_records_observation() -> None:
    state = ReActState(
        user_message="Echo hello.",
        decision=ReActDecision(
            action=ReActAction.CALL_TOOL,
            tool_call=ToolCall(tool_name="echo", tool_args={"text": "hello"}),
            reason="Need tool output.",
        ),
    )
    executor = ReActToolExecutor(make_registry())

    update = executor.execute(state)

    assert update["current_step"] == 1
    assert update["observations"][0].tool_name == "echo"
    assert update["observations"][0].success is True
    assert update["observations"][0].data == {"text": "hello"}
    assert "stop_reason" not in update


def test_react_tool_executor_stops_on_final_answer() -> None:
    state = ReActState(
        user_message="Echo hello.",
        decision=ReActDecision(
            action=ReActAction.FINAL_ANSWER,
            final_answer="Done.",
            reason="Ready.",
        ),
    )
    executor = ReActToolExecutor(make_registry())

    update = executor.execute(state)

    assert update["final_answer"] == "Done."
    assert update["stop_reason"] is StopReason.COMPLETED
    assert update["current_step"] == 1


def test_react_tool_executor_marks_approval_required() -> None:
    state = ReActState(
        user_message="Save draft.",
        decision=ReActDecision(
            action=ReActAction.CALL_TOOL,
            tool_call=ToolCall(tool_name="echo", tool_args={"text": "hello"}),
            reason="Need to save.",
        ),
    )
    executor = ReActToolExecutor(
        make_registry(
            permission=ToolPermission.REQUIRE_APPROVAL,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        )
    )

    update = executor.execute(state)

    assert update["stop_reason"] is StopReason.APPROVAL_REQUIRED
    assert update["observations"][0].success is False
    assert update["observations"][0].error["code"] == "permission_denied"


def test_react_tool_executor_marks_tool_error_for_missing_tool() -> None:
    state = ReActState(
        user_message="Call unknown.",
        decision=ReActDecision(
            action=ReActAction.CALL_TOOL,
            tool_call=ToolCall(tool_name="missing", tool_args={}),
            reason="Need tool.",
        ),
    )
    executor = ReActToolExecutor(make_registry())

    update = executor.execute(state)

    assert update["stop_reason"] is StopReason.TOOL_ERROR
    assert update["observations"][0].tool_name == "missing"
    assert update["observations"][0].error["code"] == "tool_not_found"
    assert update["observations"][0].error["recoverable"] is False


def test_react_tool_executor_allows_recoverable_validation_error_to_continue() -> None:
    state = ReActState(
        user_message="Echo hello.",
        decision=ReActDecision(
            action=ReActAction.CALL_TOOL,
            tool_call=ToolCall(tool_name="echo", tool_args={"wrong": "hello"}),
            reason="Need tool.",
        ),
    )
    executor = ReActToolExecutor(make_registry())

    update = executor.execute(state)

    assert update["current_step"] == 1
    assert "stop_reason" not in update
    assert update["observations"][0].success is False
    assert update["observations"][0].error["code"] == "validation_error"
    assert update["observations"][0].error["recoverable"] is True


def test_react_tool_executor_blocks_tool_outside_allowlist() -> None:
    state = ReActState(
        user_message="Echo hello.",
        decision=ReActDecision(
            action=ReActAction.CALL_TOOL,
            tool_call=ToolCall(tool_name="echo", tool_args={"text": "hello"}),
            reason="Need tool.",
        ),
    )
    executor = ReActToolExecutor(make_registry(), allowed_tool_names={"other_tool"})

    update = executor.execute(state)

    assert update["stop_reason"] is StopReason.TOOL_ERROR
    assert update["observations"][0].success is False
    assert update["observations"][0].error["code"] == "permission_denied"
    assert update["observations"][0].error["recoverable"] is False


def test_react_tool_executor_marks_model_error_without_decision() -> None:
    executor = ReActToolExecutor(make_registry())

    update = executor.execute(ReActState(user_message="No decision."))

    assert update["stop_reason"] is StopReason.MODEL_ERROR
    assert update["current_step"] == 1
