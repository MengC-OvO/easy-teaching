import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import ReActState
from app.services import EduFlowStore, ModelProviderError
from app.tools import build_default_tool_registry
from app.workflows import build_react_graph


def run_react_graph_smoke_test() -> None:
    store = EduFlowStore()
    store.initialize()
    registry = build_default_tool_registry(store)
    graph = build_react_graph(registry=registry)

    result = graph.invoke(
        ReActState(
            user_message=(
                "Before answering, call get_class_profile with class_id kangaroo-room. "
                "Then give a short activity planning summary for the teacher."
            ),
            max_steps=4,
        )
    )
    final_state = ReActState.model_validate(result)

    print("REACT_GRAPH_SMOKE_OK")
    print(f"stop_reason={final_state.stop_reason.value}")
    print(f"current_step={final_state.current_step}")
    print(f"observation_count={len(final_state.observations)}")
    print(f"final_answer={final_state.final_answer!r}")
    for index, observation in enumerate(final_state.observations, start=1):
        print(
            "observation_"
            f"{index}=tool:{observation.tool_name},success:{observation.success}"
        )


def main() -> int:
    try:
        run_react_graph_smoke_test()
    except ModelProviderError as error:
        print("REACT_GRAPH_SMOKE_FAILED")
        print(error.to_dict())
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
