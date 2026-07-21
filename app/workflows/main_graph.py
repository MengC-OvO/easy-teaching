from typing import Any, Dict, Mapping, Optional, Protocol, Union

from langgraph.graph import END, StateGraph

from app.agents import IntentRouter
from app.schemas import (
    Approval,
    ApprovalStatus,
    Draft,
    GraphError,
    GraphState,
    Intent,
    IntentRouteResult,
    ReActState,
    RiskLevel,
    StopReason,
    TraceEvent,
    WorkflowStatus,
)
from app.services import ModelProviderError
from app.workflows.policy_rag_graph import build_policy_rag_graph
from app.workflows.react_graph import build_react_graph


GraphStateInput = Union[GraphState, Mapping[str, Any]]


class RouterProtocol(Protocol):
    def route(self, user_message: str) -> IntentRouteResult:
        ...


class WorkflowProtocol(Protocol):
    def invoke(self, state: ReActState):
        ...


class GraphWorkflowProtocol(Protocol):
    def invoke(self, state: GraphState):
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


def build_planning_react_node(planning_workflow: WorkflowProtocol):
    def planning_react_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        result = planning_workflow.invoke(
            ReActState(
                user_message=current_state.user_message,
                max_steps=7,
            )
        )
        react_state = ReActState.model_validate(result)

        trace = [
            TraceEvent(
                step="planning_react",
                message="Activity planning ReAct workflow completed.",
                metadata={
                    "stop_reason": react_state.stop_reason.value,
                    "current_step": react_state.current_step,
                    "observations": [
                        {
                            "tool_name": observation.tool_name,
                            "success": observation.success,
                            "error_code": (
                                observation.error.get("code")
                                if observation.error
                                else None
                            ),
                        }
                        for observation in react_state.observations
                    ],
                },
            )
        ]

        if react_state.stop_reason is StopReason.COMPLETED:
            return {
                "workflow_status": WorkflowStatus.COMPLETED,
                "draft": Draft(
                    title="Activity planning draft",
                    content=react_state.final_answer or "",
                    is_draft=True,
                ),
                "trace": trace,
            }

        if react_state.stop_reason is StopReason.APPROVAL_REQUIRED:
            return {
                "workflow_status": WorkflowStatus.WAITING_FOR_APPROVAL,
                "approval": Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason="A controlled write tool requires teacher approval.",
                ),
                "trace": trace,
            }

        return {
            "workflow_status": WorkflowStatus.FAILED,
            "errors": [
                GraphError(
                    code=react_state.stop_reason.value,
                    message="Activity planning ReAct workflow stopped before completion.",
                    recoverable=react_state.stop_reason
                    in {
                        StopReason.MAX_STEPS_REACHED,
                        StopReason.TOOL_ERROR,
                        StopReason.MODEL_ERROR,
                    },
                )
            ],
            "trace": trace,
        }

    return planning_react_node


def build_policy_rag_workflow_node(policy_workflow: GraphWorkflowProtocol):
    def policy_rag_workflow_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        result = policy_workflow.invoke(current_state)
        result_state = _state_from_input(result)
        return {
            "needs_clarification": result_state.needs_clarification,
            "clarification_question": result_state.clarification_question,
            "workflow_status": result_state.workflow_status,
            "draft": result_state.draft,
            "citations": result_state.citations[len(current_state.citations) :],
            "approval": result_state.approval,
            "trace": result_state.trace[len(current_state.trace) :],
            "errors": result_state.errors[len(current_state.errors) :],
            "safety_flags": result_state.safety_flags[len(current_state.safety_flags) :],
        }

    return policy_rag_workflow_node


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


def build_main_graph(
    router: Optional[RouterProtocol] = None,
    planning_workflow: Optional[WorkflowProtocol] = None,
    policy_workflow: Optional[GraphWorkflowProtocol] = None,
):
    resolved_router = router or IntentRouter()
    resolved_planning_workflow = planning_workflow or build_react_graph(
        allowed_tool_names={
            "get_class_profile",
            "retrieve_policy_evidence",
            "check_activity_safety",
            "align_to_eylf_outcomes",
            "save_draft",
        }
    )
    resolved_policy_workflow = policy_workflow or build_policy_rag_graph()
    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node("intent_router", build_intent_router_node(resolved_router))
    graph.add_node(
        "planning_react",
        build_planning_react_node(resolved_planning_workflow),
    )
    graph.add_node("documentation_placeholder", documentation_placeholder)
    graph.add_node("policy_rag", build_policy_rag_workflow_node(resolved_policy_workflow))
    graph.add_node("family_placeholder", family_placeholder)
    graph.add_node("clarification_placeholder", clarification_placeholder)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "planning": "planning_react",
            "documentation": "documentation_placeholder",
            "policy": "policy_rag",
            "family": "family_placeholder",
            "clarification": "clarification_placeholder",
            "end": END,
        },
    )
    graph.add_edge("planning_react", END)
    graph.add_edge("documentation_placeholder", END)
    graph.add_edge("policy_rag", END)
    graph.add_edge("family_placeholder", END)
    graph.add_edge("clarification_placeholder", END)
    return graph.compile()


main_graph = build_main_graph()
