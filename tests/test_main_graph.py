from app.schemas import GraphState, WorkflowStatus
from app.workflows import build_main_graph


def test_main_graph_runs_minimal_day1_flow() -> None:
    graph = build_main_graph()
    initial_state = GraphState(
        request_id="req-graph-001",
        session_id="session-001",
        user_message="Plan an outdoor activity.",
    )

    result = graph.invoke(initial_state)
    final_state = GraphState.model_validate(result)

    assert final_state.workflow_status is WorkflowStatus.ROUTED
    assert [event.step for event in final_state.trace] == [
        "initialize",
        "route_placeholder",
    ]


def test_main_graph_preserves_core_request_fields() -> None:
    graph = build_main_graph()

    result = graph.invoke(
        {
            "request_id": "req-graph-002",
            "session_id": "session-002",
            "user_message": "Write a learning story draft.",
        }
    )
    final_state = GraphState.model_validate(result)

    assert final_state.request_id == "req-graph-002"
    assert final_state.session_id == "session-002"
    assert final_state.user_message == "Write a learning story draft."
