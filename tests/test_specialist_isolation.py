from typing import List

import pytest
from pydantic import BaseModel

from app.schemas import (
    ForbiddenSpecialistAction,
    ReActAction,
    ReActDecision,
    ReActState,
    RiskLevel,
    SpecialistInput,
    SpecialistKind,
    SpecialistPermissionDenied,
    StopReason,
    ToolCall,
    WorkflowStatus,
    get_specialist_permission,
)
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolPermission,
    ToolRegistry,
    ToolResult,
)
from app.workflows import (
    build_documentation_workflow,
    build_family_workflow,
    build_main_graph,
    build_planning_workflow,
    build_policy_rag_graph,
)


class EmptyInput(BaseModel):
    pass


class EmptyOutput(BaseModel):
    ok: bool


def successful_read(_input: BaseModel) -> ToolResult:
    return ToolResult.ok(
        data={"ok": True},
        risk_level=RiskLevel.L0_READ_ONLY,
    )


def registry_with_planning_tool() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_class_profile",
            description="Return a synthetic class profile.",
            category=ToolCategory.CLASS_PROFILE,
            input_model=EmptyInput,
            output_model=EmptyOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            handler=successful_read,
        )
    )
    return registry


class FixedDecisionAgent:
    def __init__(self, decision: ReActDecision) -> None:
        self.decision = decision
        self.seen_tool_names: List[str] = []

    def decide(
        self,
        state: ReActState,
        available_tools: List[ToolDefinition],
    ) -> ReActDecision:
        self.seen_tool_names = [tool.name for tool in available_tools]
        return self.decision


def planning_input() -> SpecialistInput:
    return SpecialistInput(
        specialist=SpecialistKind.PLANNING,
        request_id="req-isolation",
        session_id="session-isolation",
        user_message="Plan a synthetic classroom activity.",
    )


def tool_decision(tool_name: str) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.CALL_TOOL,
        reason="Request a tool.",
        tool_call=ToolCall(tool_name=tool_name, tool_args={}),
    )


def test_planning_runtime_blocks_unlisted_tool_and_traces_reason() -> None:
    agent = FixedDecisionAgent(tool_decision("send_family_message"))
    workflow = build_planning_workflow(
        agent=agent,
        registry=ToolRegistry(),
    )

    result = workflow.invoke(planning_input())

    assert "send_family_message" not in agent.seen_tool_names
    assert result.status is WorkflowStatus.FAILED
    assert result.errors[0].code == StopReason.TOOL_ERROR.value
    assert result.trace[0].metadata["observations"] == [
        {
            "tool_name": "send_family_message",
            "success": False,
            "error_code": "permission_denied",
        }
    ]


def test_planning_runtime_stops_at_reduced_step_budget() -> None:
    agent = FixedDecisionAgent(tool_decision("get_class_profile"))
    workflow = build_planning_workflow(
        agent=agent,
        registry=registry_with_planning_tool(),
        allowed_tool_names={"get_class_profile"},
        max_steps=1,
    )

    result = workflow.invoke(planning_input())

    assert result.status is WorkflowStatus.FAILED
    assert result.errors[0].code == StopReason.MAX_STEPS_REACHED.value
    assert result.trace[0].metadata["current_step"] == 1


def test_policy_cannot_use_planning_write_tool() -> None:
    permission = get_specialist_permission(SpecialistKind.POLICY)

    with pytest.raises(SpecialistPermissionDenied, match="save_draft"):
        permission.require_tool("save_draft")


def test_documentation_cannot_release_raw_pii() -> None:
    workflow = build_documentation_workflow()

    with pytest.raises(SpecialistPermissionDenied, match="raw_pii_output"):
        workflow.permission.require_action(
            ForbiddenSpecialistAction.RAW_PII_OUTPUT
        )


def test_family_cannot_send_real_world_message() -> None:
    workflow = build_family_workflow()

    with pytest.raises(SpecialistPermissionDenied, match="real_world_send"):
        workflow.permission.require_action(
            ForbiddenSpecialistAction.REAL_WORLD_SEND
        )


def test_policy_fixed_workflow_exposes_no_function_calling_tools() -> None:
    class UnusedPolicyService:
        def answer(self, question: str, *, conversation_context: str = ""):
            raise AssertionError("The service should not run in this configuration test")

    workflow = build_policy_rag_graph(UnusedPolicyService())

    assert workflow.permission.allowed_tool_names == frozenset()
    with pytest.raises(SpecialistPermissionDenied, match="get_class_profile"):
        workflow.permission.require_tool("get_class_profile")


def test_main_graph_rejects_cross_specialist_permission_mapping() -> None:
    with pytest.raises(SpecialistPermissionDenied, match="cannot use policy"):
        build_main_graph(
            specialist_permissions={
                SpecialistKind.PLANNING: get_specialist_permission(
                    SpecialistKind.POLICY
                )
            }
        )
