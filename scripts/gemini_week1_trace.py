import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import IntentRouter, ReActAgent
from app.schemas import GraphState, ReActDecision, ReActState
from app.services import EduFlowStore, ModelProviderError
from app.tools import ToolDefinition, build_default_tool_registry
from app.workflows import build_main_graph, build_planning_workflow


DEFAULT_MESSAGE = """
Plan and save a synthetic outdoor sensory activity draft for the Kangaroo Room.
Use the available tools step by step:
1. Read class context with get_class_profile using class_id kangaroo-room.
2. Retrieve risk guidance with retrieve_risk_guidance using query outdoor sensory play.
3. Check the draft idea with check_activity_safety.
4. Align the draft idea to EYLF outcomes with align_to_eylf_outcomes.
5. Save the draft with save_draft. Include draft_id gemini-demo-draft-001,
   idempotency_key gemini-demo:save-draft, draft_type activity_plan,
   title Outdoor sensory walk, and synthetic draft content.
6. After the draft is saved, provide a concise final answer for the teacher.
""".strip()


class TracingReActAgent:
    def __init__(self) -> None:
        self.agent = ReActAgent()

    def decide(self, state: ReActState, available_tools: List[ToolDefinition]) -> ReActDecision:
        print("\n--- Gemini ReAct decision request ---")
        print(f"current_step={state.current_step}")
        print(f"max_steps={state.max_steps}")
        print(f"available_tools={[tool.name for tool in available_tools]}")
        if state.observations:
            print("previous_observations=")
            for index, observation in enumerate(state.observations, start=1):
                print(
                    "  "
                    f"{index}. tool={observation.tool_name}, "
                    f"success={observation.success}, "
                    f"error={observation.error}"
                )
        else:
            print("previous_observations=[]")

        decision = self.agent.decide(state, available_tools)

        print("--- Gemini ReAct decision response ---")
        print(json.dumps(decision.model_dump(mode="json"), indent=2, ensure_ascii=False))
        return decision


def build_graph(database_url: str, *, approved: bool):
    store = EduFlowStore(database_url)
    store.initialize()
    registry = build_default_tool_registry(store)
    planning_workflow = build_planning_workflow(
        agent=TracingReActAgent(),
        registry=registry,
        allowed_tool_names={
            "get_class_profile",
            "retrieve_risk_guidance",
            "check_activity_safety",
            "align_to_eylf_outcomes",
            "save_draft",
        },
        approved=approved,
        required_skill_name=None,
    )
    return build_main_graph(IntentRouter(), planning_workflow=planning_workflow)


def print_final_state(state: GraphState) -> None:
    print("\n=== Final GraphState ===")
    print(f"request_id={state.request_id}")
    print(f"session_id={state.session_id}")
    print(f"intent={state.intent.value}")
    print(f"workflow_status={state.workflow_status.value}")
    print(f"approval_status={state.approval.status.value}")
    print(f"approval_reason={state.approval.reason}")

    if state.draft:
        print("\n=== Draft ===")
        print(f"title={state.draft.title}")
        print(f"is_draft={state.draft.is_draft}")
        print(f"content={state.draft.content}")

    if state.errors:
        print("\n=== Errors ===")
        for error in state.errors:
            print(json.dumps(error.model_dump(mode="json"), indent=2, ensure_ascii=False))

    print("\n=== Trace ===")
    for event in state.trace:
        print(f"- {event.step}: {event.message}")
        if event.metadata:
            print(json.dumps(event.metadata, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real Gemini Week1 trace through IntentRouter + ReAct tools."
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="Teacher request to send through the graph.",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Do not pre-approve L2 tools; save_draft should stop at approval_required.",
    )
    args = parser.parse_args()

    approved = not args.no_approval

    print("GEMINI_WEEK1_TRACE_START")
    print(f"approved={approved}")
    print(f"message={args.message!r}")

    try:
        with tempfile.TemporaryDirectory() as directory:
            database_url = f"sqlite:///{Path(directory) / 'gemini-week1-trace.sqlite3'}"
            graph = build_graph(database_url, approved=approved)
            result = graph.invoke(
                GraphState(
                    request_id="gemini-week1-trace",
                    session_id="gemini-demo-session",
                    user_message=args.message,
                )
            )
            final_state = GraphState.model_validate(result)
    except ModelProviderError as error:
        print("GEMINI_WEEK1_TRACE_FAILED")
        print(json.dumps(error.to_dict(), indent=2, ensure_ascii=False))
        return 1

    print_final_state(final_state)
    print("\nGEMINI_WEEK1_TRACE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
