from typing import Any, Dict, Mapping, Optional, Protocol, Union

from langgraph.graph import END, StateGraph

from app.agents import IntentRouter
from app.schemas import GraphError, GraphState, Intent, IntentRouteResult, TraceEvent, WorkflowStatus
from app.services import ModelProviderError


GraphStateInput = Union[GraphState, Mapping[str, Any]]


class RouterProtocol(Protocol):
    def route(self, user_message: str) -> IntentRouteResult:
        ...


def _state_from_input(state: GraphStateInput) -> GraphState:
    if isinstance(state, GraphState):
        return state
    return GraphState.model_validate(state)


def initialize(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "trace": [
            TraceEvent(
                step="initialize",
                message="Initialized EduFlow graph state.",
            )
        ]
    }


def build_intent_router_node(router: RouterProtocol):
    def intent_router_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        try:
            route_result = router.route(current_state.user_message)
        except ModelProviderError as error:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "errors": [
                    GraphError(
                        code=error.code.value,
                        message=error.message,
                        recoverable=error.recoverable,
                    )
                ],
                "trace": [
                    TraceEvent(
                        step="intent_router",
                        message="Intent routing failed.",
                        metadata=error.to_dict(),
                    )
                ],
            }

        return {
            "intent": route_result.intent,
            "needs_clarification": route_result.needs_clarification,
            "clarification_question": route_result.clarification_question,
            "workflow_status": WorkflowStatus.ROUTED,
            "trace": [
                TraceEvent(
                    step="intent_router",
                    message="Intent routing completed.",
                    metadata=route_result.model_dump(),
                )
            ],
        }

    return intent_router_node


def route_by_intent(state: GraphStateInput) -> str:
    current_state = _state_from_input(state)
    if current_state.workflow_status is WorkflowStatus.FAILED:
        return "end"
    if current_state.needs_clarification or current_state.intent is Intent.UNKNOWN:
        return "clarification"
    if current_state.intent is Intent.ACTIVITY_PLANNING:
        return "planning"
    if current_state.intent is Intent.LEARNING_RECORD:
        return "documentation"
    if current_state.intent is Intent.POLICY_QA:
        return "policy"
    if current_state.intent is Intent.FAMILY_COMMUNICATION:
        return "family"
    return "clarification"


def planning_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "trace": [
            TraceEvent(
                step="planning_placeholder",
                message="Routed to activity planning workflow placeholder.",
            )
        ]
    }


def documentation_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "trace": [
            TraceEvent(
                step="documentation_placeholder",
                message="Routed to learning record workflow placeholder.",
            )
        ]
    }


def policy_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "trace": [
            TraceEvent(
                step="policy_placeholder",
                message="Routed to policy QA workflow placeholder.",
            )
        ]
    }


def family_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "trace": [
            TraceEvent(
                step="family_placeholder",
                message="Routed to family communication workflow placeholder.",
            )
        ]
    }


def clarification_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "trace": [
            TraceEvent(
                step="clarification_placeholder",
                message="Routed to clarification placeholder.",
            )
        ]
    }


def build_main_graph(router: Optional[RouterProtocol] = None):
    resolved_router = router or IntentRouter()
    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node("intent_router", build_intent_router_node(resolved_router))
    graph.add_node("planning_placeholder", planning_placeholder)
    graph.add_node("documentation_placeholder", documentation_placeholder)
    graph.add_node("policy_placeholder", policy_placeholder)
    graph.add_node("family_placeholder", family_placeholder)
    graph.add_node("clarification_placeholder", clarification_placeholder)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "planning": "planning_placeholder",
            "documentation": "documentation_placeholder",
            "policy": "policy_placeholder",
            "family": "family_placeholder",
            "clarification": "clarification_placeholder",
            "end": END,
        },
    )
    graph.add_edge("planning_placeholder", END)
    graph.add_edge("documentation_placeholder", END)
    graph.add_edge("policy_placeholder", END)
    graph.add_edge("family_placeholder", END)
    graph.add_edge("clarification_placeholder", END)
    return graph.compile()


main_graph = build_main_graph()
