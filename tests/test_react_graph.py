from typing import List

from pydantic import BaseModel

from app.schemas import (
    ReActAction,
    ReActDecision,
    ReActState,
    RiskLevel,
    StopReason,
    ToolCall,
)
from app.tools import ToolCategory, ToolDefinition, ToolPermission, ToolRegistry, ToolResult
from app.workflows import build_react_graph


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


class SequencedAgent:
    def __init__(self, decisions: List[ReActDecision]) -> None:
        self.decisions = decisions
        self.calls = 0
        self.seen_tool_names: List[str] = []

    def decide(self, state: ReActState, available_tools: List[ToolDefinition]) -> ReActDecision:
        self.calls += 1
        self.seen_tool_names = [tool.name for tool in available_tools]
        return self.decisions.pop(0)


def call_echo_decision(text: str = "hello") -> ReActDecision:
    return ReActDecision(
        action=ReActAction.CALL_TOOL,
        reason="Need tool output.",
        tool_call=ToolCall(tool_name="echo", tool_args={"text": text}),
    )


def call_echo_with_args(tool_args) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.CALL_TOOL,
        reason="Need tool output.",
        tool_call=ToolCall(tool_name="echo", tool_args=tool_args),
    )


def final_answer_decision(answer: str = "Done.") -> ReActDecision:
    return ReActDecision(
        action=ReActAction.FINAL_ANSWER,
        reason="Ready to answer.",
        final_answer=answer,
    )


def test_react_graph_calls_tool_then_finishes_with_final_answer() -> None:
    agent = SequencedAgent(
        [
            call_echo_decision("hello"),
            final_answer_decision("The tool said hello."),
        ]
    )
    graph = build_react_graph(agent=agent, registry=make_registry())

    result = graph.invoke(ReActState(user_message="Echo hello.", max_steps=3))

    assert result["final_answer"] == "The tool said hello."
    assert result["stop_reason"] is StopReason.COMPLETED
    assert result["current_step"] == 2
    assert result["observations"][0].data == {"text": "hello"}
    assert agent.calls == 2


def test_react_graph_stops_when_step_budget_is_used() -> None:
    agent = SequencedAgent([call_echo_decision("one")])
    graph = build_react_graph(agent=agent, registry=make_registry())

    result = graph.invoke(ReActState(user_message="Keep calling.", max_steps=1))

    assert result["stop_reason"] is StopReason.MAX_STEPS_REACHED
    assert result["current_step"] == 1
    assert result["observations"][0].data == {"text": "one"}
    assert agent.calls == 1


def test_react_graph_continues_after_recoverable_tool_error() -> None:
    agent = SequencedAgent(
        [
            call_echo_with_args({"wrong": "hello"}),
            call_echo_decision("hello"),
            final_answer_decision("Recovered."),
        ]
    )
    graph = build_react_graph(agent=agent, registry=make_registry())

    result = graph.invoke(ReActState(user_message="Echo hello.", max_steps=4))

    assert result["stop_reason"] is StopReason.COMPLETED
    assert result["final_answer"] == "Recovered."
    assert result["current_step"] == 3
    assert result["observations"][0].success is False
    assert result["observations"][0].error["code"] == "validation_error"
    assert result["observations"][1].success is True
    assert result["observations"][1].data == {"text": "hello"}
    assert agent.calls == 3


def test_react_graph_stops_when_tool_requires_approval() -> None:
    agent = SequencedAgent([call_echo_decision("draft")])
    graph = build_react_graph(
        agent=agent,
        registry=make_registry(
            permission=ToolPermission.REQUIRE_APPROVAL,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        ),
    )

    result = graph.invoke(ReActState(user_message="Save a draft.", max_steps=3))

    assert result["stop_reason"] is StopReason.APPROVAL_REQUIRED
    assert result["current_step"] == 1
    assert result["observations"][0].success is False
    assert result["observations"][0].error["code"] == "permission_denied"


def test_react_graph_exposes_and_enforces_tool_allowlist() -> None:
    agent = SequencedAgent([call_echo_decision("hello")])
    graph = build_react_graph(
        agent=agent,
        registry=make_registry(),
        allowed_tool_names={"other_tool"},
    )

    result = graph.invoke(ReActState(user_message="Echo hello.", max_steps=3))

    assert agent.seen_tool_names == []
    assert result["stop_reason"] is StopReason.TOOL_ERROR
    assert result["observations"][0].error["code"] == "permission_denied"
    assert result["observations"][0].error["recoverable"] is False


def test_react_graph_stops_on_model_provider_error() -> None:
    class FailingAgent:
        def decide(
            self,
            state: ReActState,
            available_tools: List[ToolDefinition],
        ) -> ReActDecision:
            from app.services import ModelProviderError

            raise ModelProviderError("model failed")

    graph = build_react_graph(agent=FailingAgent(), registry=make_registry())

    result = graph.invoke(ReActState(user_message="Do something.", max_steps=3))

    assert result["stop_reason"] is StopReason.MODEL_ERROR
    assert result["current_step"] == 0
