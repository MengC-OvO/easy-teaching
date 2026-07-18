from typing import List

from app.schemas import (
    ApprovalStatus,
    GraphState,
    Intent,
    IntentRouteResult,
    ReActAction,
    ReActDecision,
    ReActState,
    StopReason,
    ToolCall,
    WorkflowStatus,
)
from app.services import EduFlowStore
from app.tools import ToolDefinition, build_mock_tool_registry
from app.workflows import build_main_graph, build_react_graph


class StubRouter:
    def route(self, user_message: str) -> IntentRouteResult:
        return IntentRouteResult(
            intent=Intent.ACTIVITY_PLANNING,
            confidence=0.95,
            reason="The request asks for an activity plan.",
        )


class SequencedPlanningAgent:
    def __init__(self, decisions: List[ReActDecision]) -> None:
        self.decisions = decisions
        self.calls = 0

    def decide(self, state: ReActState, available_tools: List[ToolDefinition]) -> ReActDecision:
        self.calls += 1
        return self.decisions.pop(0)


def call_tool(name: str, args) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.CALL_TOOL,
        reason=f"Need to call {name}.",
        tool_call=ToolCall(tool_name=name, tool_args=args),
    )


def final_answer(content: str) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.FINAL_ANSWER,
        reason="The draft is ready.",
        final_answer=content,
    )


def make_store(tmp_path) -> EduFlowStore:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'week1-chain.sqlite3'}")
    store.initialize()
    return store


def build_planning_chain(tmp_path, *, approved: bool, agent: SequencedPlanningAgent):
    registry = build_mock_tool_registry(make_store(tmp_path))
    planning_workflow = build_react_graph(
        agent=agent,
        registry=registry,
        allowed_tool_names={
            "get_class_profile",
            "search_policy_index",
            "save_draft",
        },
        approved=approved,
    )
    return build_main_graph(StubRouter(), planning_workflow=planning_workflow)


def planning_decisions(include_final_answer: bool = True) -> List[ReActDecision]:
    decisions = [
        call_tool("get_class_profile", {"class_id": "kangaroo-room"}),
        call_tool("search_policy_index", {"query": "program"}),
        call_tool(
            "save_draft",
            {
                "draft_id": "draft-week1-001",
                "idempotency_key": "req-week1:save-draft",
                "draft_type": "activity_plan",
                "title": "Outdoor sensory walk",
                "content": "Synthetic activity plan draft.",
            },
        ),
    ]
    if include_final_answer:
        decisions.append(final_answer("Synthetic activity plan draft."))
    return decisions


def test_week1_chain_routes_to_react_tools_and_waits_for_approval(tmp_path) -> None:
    agent = SequencedPlanningAgent(planning_decisions(include_final_answer=False))
    graph = build_planning_chain(tmp_path, approved=False, agent=agent)

    final_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-week1",
                session_id="session-week1",
                user_message="Plan and save an outdoor sensory activity.",
            )
        )
    )

    assert final_state.workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert final_state.approval.status is ApprovalStatus.REQUIRED
    assert final_state.trace[-1].step == "planning_react"
    assert final_state.trace[-1].metadata["stop_reason"] == "approval_required"
    assert [item["tool_name"] for item in final_state.trace[-1].metadata["observations"]] == [
        "get_class_profile",
        "search_policy_index",
        "save_draft",
    ]
    assert agent.calls == 3


def test_week1_chain_routes_to_react_tools_and_saves_draft_when_approved(tmp_path) -> None:
    agent = SequencedPlanningAgent(planning_decisions())
    graph = build_planning_chain(tmp_path, approved=True, agent=agent)

    final_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-week1-approved",
                session_id="session-week1",
                user_message="Plan and save an outdoor sensory activity.",
            )
        )
    )

    assert final_state.workflow_status is WorkflowStatus.COMPLETED
    assert final_state.draft is not None
    assert final_state.draft.content == "Synthetic activity plan draft."
    assert final_state.trace[-1].metadata["stop_reason"] == StopReason.COMPLETED.value
    assert final_state.trace[-1].metadata["observations"][-1] == {
        "tool_name": "save_draft",
        "success": True,
        "error_code": None,
    }
    assert agent.calls == 4
