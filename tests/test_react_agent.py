from pydantic import BaseModel

from app.agents import REACT_AGENT_SYSTEM_PROMPT, ReActAgent
from app.schemas import (
    Observation,
    ReActAction,
    ReActDecision,
    ReActState,
    ToolCall,
)
from app.services import ModelResponse
from app.tools import ToolCategory, ToolDefinition, ToolPermission, ToolResult
from app.schemas import RiskLevel


class ToolInput(BaseModel):
    class_id: str


class ToolOutput(BaseModel):
    name: str


def tool_handler(input_data: BaseModel) -> ToolResult:
    return ToolResult.ok(data={"name": "Kangaroo Room"}, risk_level=RiskLevel.L0_READ_ONLY)


def make_tool() -> ToolDefinition:
    return ToolDefinition(
        name="get_class_profile",
        description="Read a synthetic class profile.",
        category=ToolCategory.CLASS_PROFILE,
        input_model=ToolInput,
        output_model=ToolOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=tool_handler,
    )


class StubReActProvider:
    def __init__(self, decision: ReActDecision) -> None:
        self.decision = decision
        self.messages = None
        self.response_model = None
        self.temperature = None

    def generate_structured(self, *, messages, response_model, temperature=0.0):
        self.messages = messages
        self.response_model = response_model
        self.temperature = temperature
        return ModelResponse(
            content=self.decision.model_dump_json(),
            structured=self.decision,
        )


def test_react_agent_prompt_describes_allowed_actions() -> None:
    assert "call_tool" in REACT_AGENT_SYSTEM_PROMPT
    assert "final_answer" in REACT_AGENT_SYSTEM_PROMPT
    assert "Do not invent tool names" in REACT_AGENT_SYSTEM_PROMPT


def test_react_agent_requests_structured_decision() -> None:
    decision = ReActDecision(
        action=ReActAction.CALL_TOOL,
        tool_call=ToolCall(
            tool_name="get_class_profile",
            tool_args={"class_id": "kangaroo-room"},
        ),
        reason="Need class context.",
    )
    provider = StubReActProvider(decision)
    agent = ReActAgent(provider)

    result = agent.decide(
        ReActState(user_message="Plan an outdoor activity."),
        [make_tool()],
    )

    assert result == decision
    assert provider.response_model is ReActDecision
    assert provider.temperature == 0.0
    assert provider.messages[0].role.value == "system"
    assert "get_class_profile" in provider.messages[1].content


def test_react_agent_includes_previous_observations_in_prompt() -> None:
    decision = ReActDecision(
        action=ReActAction.FINAL_ANSWER,
        final_answer="Use outdoor sensory exploration.",
        reason="Observation gives enough context.",
    )
    provider = StubReActProvider(decision)
    agent = ReActAgent(provider)

    result = agent.decide(
        ReActState(
            user_message="Plan an outdoor activity.",
            observations=[
                Observation(
                    tool_name="get_class_profile",
                    success=True,
                    data={"name": "Kangaroo Room"},
                )
            ],
        ),
        [make_tool()],
    )

    assert result.action is ReActAction.FINAL_ANSWER
    assert "Kangaroo Room" in provider.messages[1].content
