import sys
import tempfile
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import (
    GraphState,
    Intent,
    IntentRouteResult,
    ReActAction,
    ReActDecision,
    ReActState,
    ToolCall,
)
from app.services import EduFlowStore
from app.tools import ToolDefinition, build_default_tool_registry
from app.workflows import build_main_graph, build_react_graph


class DemoRouter:
    def route(self, user_message: str) -> IntentRouteResult:
        return IntentRouteResult(
            intent=Intent.ACTIVITY_PLANNING,
            confidence=0.95,
            reason="Demo route: the request asks for an activity plan.",
        )


class DemoPlanningAgent:
    def __init__(self, decisions: List[ReActDecision]) -> None:
        self.decisions = decisions

    def decide(self, state: ReActState, available_tools: List[ToolDefinition]) -> ReActDecision:
        return self.decisions.pop(0)


def call_tool(name: str, args) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.CALL_TOOL,
        reason=f"Demo step: call {name}.",
        tool_call=ToolCall(tool_name=name, tool_args=args),
    )


def final_answer(content: str) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.FINAL_ANSWER,
        reason="Demo step: final draft is ready.",
        final_answer=content,
    )


def planning_decisions(*, include_final_answer: bool) -> List[ReActDecision]:
    decisions = [
        call_tool("get_class_profile", {"class_id": "kangaroo-room"}),
        call_tool("retrieve_risk_guidance", {"query": "outdoor sensory play", "top_k": 3}),
        call_tool(
            "check_activity_safety",
            {
                "activity_text": "Outdoor sensory walk for Kangaroo Room.",
                "age_group": "3-5",
                "class_size": 18,
            },
        ),
        call_tool(
            "align_to_eylf_outcomes",
            {
                "activity_text": (
                    "Children explore outdoor natural materials through play "
                    "and describe sounds, textures, and patterns."
                ),
                "top_k": 3,
            },
        ),
        call_tool(
            "save_draft",
            {
                "draft_id": "demo-draft-001",
                "idempotency_key": "demo-week1:save-draft",
                "draft_type": "activity_plan",
                "title": "Outdoor sensory walk",
                "content": "Draft: Outdoor sensory walk for Kangaroo Room.",
            },
        ),
    ]
    if include_final_answer:
        decisions.append(final_answer("Draft: Outdoor sensory walk for Kangaroo Room."))
    return decisions


def build_demo_graph(database_url: str, *, approved: bool, include_final_answer: bool):
    store = EduFlowStore(database_url)
    store.initialize()
    registry = build_default_tool_registry(store)
    planning_workflow = build_react_graph(
        agent=DemoPlanningAgent(
            planning_decisions(include_final_answer=include_final_answer)
        ),
        registry=registry,
        allowed_tool_names={
            "get_class_profile",
            "retrieve_risk_guidance",
            "check_activity_safety",
            "align_to_eylf_outcomes",
            "save_draft",
        },
        approved=approved,
    )
    return build_main_graph(DemoRouter(), planning_workflow=planning_workflow)


def run_case(database_url: str, *, approved: bool, include_final_answer: bool) -> GraphState:
    graph = build_demo_graph(
        database_url,
        approved=approved,
        include_final_answer=include_final_answer,
    )
    result = graph.invoke(
        GraphState(
            request_id="demo-week1",
            session_id="demo-session",
            user_message="Plan and save an outdoor sensory activity for Kangaroo Room.",
        )
    )
    return GraphState.model_validate(result)


def print_state(label: str, state: GraphState) -> None:
    print(f"\n{label}")
    print(f"intent={state.intent.value}")
    print(f"workflow_status={state.workflow_status.value}")
    print(f"approval_status={state.approval.status.value}")
    if state.draft:
        print(f"draft_title={state.draft.title}")
        print(f"draft_content={state.draft.content}")

    print(f"trace_steps={[event.step for event in state.trace]}")
    planning_trace = state.trace[-1]
    print(f"react_stop_reason={planning_trace.metadata.get('stop_reason')}")
    print("tools_called=")
    for item in planning_trace.metadata.get("observations", []):
        print(
            "  - "
            f"{item['tool_name']}: success={item['success']}, "
            f"error_code={item['error_code']}"
        )


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        database_url = f"sqlite:///{Path(directory) / 'week1-demo.sqlite3'}"
        waiting_state = run_case(
            database_url,
            approved=False,
            include_final_answer=False,
        )
        completed_state = run_case(
            database_url,
            approved=True,
            include_final_answer=True,
        )

    print("WEEK1_DEMO_OK")
    print_state("CASE 1: approval required", waiting_state)
    print_state("CASE 2: approved and completed", completed_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
