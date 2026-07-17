from app.schemas import GraphState, Intent, IntentRouteResult, WorkflowStatus
from app.services import ModelTimeoutError
from app.workflows import build_main_graph


class StubRouter:
    def __init__(self, result: IntentRouteResult) -> None:
        self.result = result
        self.user_message = None

    def route(self, user_message: str) -> IntentRouteResult:
        self.user_message = user_message
        return self.result


class FailingRouter:
    def route(self, user_message: str) -> IntentRouteResult:
        raise ModelTimeoutError("router timed out")


def test_main_graph_runs_intent_router_node() -> None:
    router = StubRouter(
        IntentRouteResult(
            intent=Intent.ACTIVITY_PLANNING,
            confidence=0.9,
            reason="The request asks for an activity plan.",
        )
    )
    graph = build_main_graph(router)
    initial_state = GraphState(
        request_id="req-graph-001",
        session_id="session-001",
        user_message="Plan an outdoor activity.",
    )

    result = graph.invoke(initial_state)
    final_state = GraphState.model_validate(result)

    assert final_state.workflow_status is WorkflowStatus.ROUTED
    assert final_state.intent is Intent.ACTIVITY_PLANNING
    assert router.user_message == "Plan an outdoor activity."
    assert [event.step for event in final_state.trace] == [
        "initialize",
        "intent_router",
        "planning_placeholder",
    ]


def test_main_graph_preserves_core_request_fields() -> None:
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.LEARNING_RECORD,
                confidence=0.9,
                reason="The request asks for documentation.",
            )
        )
    )

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


def test_main_graph_records_router_errors() -> None:
    graph = build_main_graph(FailingRouter())

    result = graph.invoke(
        {
            "request_id": "req-graph-003",
            "session_id": "session-003",
            "user_message": "Plan an activity.",
        }
    )
    final_state = GraphState.model_validate(result)

    assert final_state.workflow_status is WorkflowStatus.FAILED
    assert final_state.errors[0].code == "timeout"
    assert final_state.trace[-1].step == "intent_router"


def test_main_graph_routes_learning_record_to_documentation_placeholder() -> None:
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.LEARNING_RECORD,
                confidence=0.9,
                reason="The request asks for a learning record.",
            )
        )
    )

    final_state = GraphState.model_validate(
        graph.invoke(
            {
                "request_id": "req-doc",
                "session_id": "session-doc",
                "user_message": "Write a learning story draft.",
            }
        )
    )

    assert final_state.trace[-1].step == "documentation_placeholder"


def test_main_graph_routes_policy_qa_to_policy_placeholder() -> None:
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.POLICY_QA,
                confidence=0.88,
                reason="The request asks about policy.",
            )
        )
    )

    final_state = GraphState.model_validate(
        graph.invoke(
            {
                "request_id": "req-policy",
                "session_id": "session-policy",
                "user_message": "What does NQS QA1 require?",
            }
        )
    )

    assert final_state.trace[-1].step == "policy_placeholder"


def test_main_graph_routes_family_communication_to_family_placeholder() -> None:
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.FAMILY_COMMUNICATION,
                confidence=0.86,
                reason="The request asks for a family message.",
            )
        )
    )

    final_state = GraphState.model_validate(
        graph.invoke(
            {
                "request_id": "req-family",
                "session_id": "session-family",
                "user_message": "Draft a parent update.",
            }
        )
    )

    assert final_state.trace[-1].step == "family_placeholder"


def test_main_graph_routes_clarification_to_clarification_placeholder() -> None:
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.UNKNOWN,
                confidence=0.4,
                needs_clarification=True,
                clarification_question="Do you want an activity plan or a family message?",
                reason="The request is ambiguous.",
            )
        )
    )

    final_state = GraphState.model_validate(
        graph.invoke(
            {
                "request_id": "req-clarify",
                "session_id": "session-clarify",
                "user_message": "Can you write something for tomorrow?",
            }
        )
    )

    assert final_state.needs_clarification is True
    assert final_state.clarification_question == "Do you want an activity plan or a family message?"
    assert final_state.trace[-1].step == "clarification_placeholder"
