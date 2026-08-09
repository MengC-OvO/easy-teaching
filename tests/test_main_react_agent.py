import asyncio

from pydantic import BaseModel

from app.agents import MAIN_REACT_SYSTEM_PROMPT, MainReActAgent
from app.schemas import (
    CapabilityCall,
    CapabilityObservation,
    CapabilitySource,
    MainDecision,
    ObservationStatus,
    RiskLevel,
)
from app.services import ModelResponse
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolResult,
)


class EmptyInput(BaseModel):
    pass


class EmptyOutput(BaseModel):
    value: str


def make_tool() -> ToolDefinition:
    return ToolDefinition(
        name="search_eylf",
        description="Search EYLF evidence.",
        category=ToolCategory.POLICY,
        input_model=EmptyInput,
        output_model=EmptyOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.INTERNAL,
        parallel_safe=True,
        handler=lambda _: ToolResult.ok(
            data={"value": "ok"},
            risk_level=RiskLevel.L0_READ_ONLY,
        ),
    )


class StubProvider:
    def __init__(self, decision: MainDecision) -> None:
        self.decision = decision
        self.messages = None
        self.response_model = None

    def generate_structured(self, *, messages, response_model, temperature=0.0):
        self.messages = messages
        self.response_model = response_model
        return ModelResponse(
            content=self.decision.model_dump_json(),
            structured=self.decision,
        )


def test_main_agent_requests_current_structured_decision() -> None:
    provider = StubProvider(
        MainDecision(
            reason="需要内部证据。",
            tool_calls=[
                CapabilityCall(
                    name="search_eylf",
                    result_key="eylf",
                )
            ],
        )
    )
    agent = MainReActAgent(provider)

    result = asyncio.run(
        agent.decide(
            user_message="查询Outcome 4。",
            conversation_context="",
            observations={},
            available_tools=[make_tool()],
            available_workers=[],
            current_step=0,
            max_steps=8,
        )
    )

    assert result.tool_calls[0].name == "search_eylf"
    assert provider.response_model is MainDecision
    assert "Never" in provider.messages[0].content
    assert "search_eylf" in provider.messages[1].content


def test_main_agent_prompt_contains_observation_and_workers() -> None:
    provider = StubProvider(
        MainDecision(reason="信息足够。", final_answer="活动草稿。")
    )
    agent = MainReActAgent(provider)

    asyncio.run(
        agent.decide(
            user_message="生成活动。",
            conversation_context="老师偏好户外活动。",
            observations={
                "weather": CapabilityObservation(
                    result_key="weather",
                    capability_name="get_weather",
                    source_kind=CapabilitySource.MCP,
                    status=ObservationStatus.COMPLETED,
                    data={"condition": "sunny"},
                )
            },
            available_tools=[],
            available_workers=[
                {"name": "external_research_worker", "description": "公开研究"}
            ],
            current_step=2,
            max_steps=8,
        )
    )

    prompt = provider.messages[1].content
    assert "weather" in prompt
    assert "sunny" in prompt
    assert "external_research_worker" in prompt
    assert "老师偏好户外活动" in prompt


def test_prompt_limits_workers_to_independent_deep_tasks() -> None:
    assert "at least two independent deep research tasks" in MAIN_REACT_SYSTEM_PROMPT
    assert "single deep task stays in Main" in MAIN_REACT_SYSTEM_PROMPT

