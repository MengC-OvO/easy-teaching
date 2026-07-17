import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import IntentRouter
from app.schemas import GraphState
from app.services import ModelProviderError
from app.workflows import build_main_graph


def run_router_smoke_test() -> None:
    router = IntentRouter()
    result = router.route("Please plan an outdoor sensory activity for preschool children.")

    print("INTENT_ROUTER_SMOKE_OK")
    print(f"intent={result.intent.value}")
    print(f"confidence={result.confidence}")
    print(f"needs_clarification={result.needs_clarification}")
    print(f"reason={result.reason}")


def run_graph_smoke_test() -> None:
    graph = build_main_graph()
    result = graph.invoke(
        GraphState(
            request_id="smoke-router-001",
            session_id="smoke-session-001",
            user_message="Please plan an outdoor sensory activity for preschool children.",
        )
    )
    final_state = GraphState.model_validate(result)

    print("INTENT_GRAPH_SMOKE_OK")
    print(f"intent={final_state.intent.value}")
    print(f"workflow_status={final_state.workflow_status.value}")
    print(f"trace_steps={[event.step for event in final_state.trace]}")


def main() -> int:
    try:
        run_router_smoke_test()
        run_graph_smoke_test()
    except ModelProviderError as error:
        print("INTENT_ROUTER_SMOKE_FAILED")
        print(error.to_dict())
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
