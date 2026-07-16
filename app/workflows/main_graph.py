from typing import Any, Dict, Mapping, Union

from langgraph.graph import END, StateGraph

from app.schemas import GraphState, TraceEvent, WorkflowStatus


GraphStateInput = Union[GraphState, Mapping[str, Any]]


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


def route_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "workflow_status": WorkflowStatus.ROUTED,
        "trace": [
            TraceEvent(
                step="route_placeholder",
                message="Routing placeholder completed without selecting a specialist workflow.",
            )
        ],
    }


def build_main_graph():
    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node("route_placeholder", route_placeholder)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "route_placeholder")
    graph.add_edge("route_placeholder", END)
    return graph.compile()


main_graph = build_main_graph()
