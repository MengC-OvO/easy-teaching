from typing import Any, Dict, List, Mapping, Optional, Protocol, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from app.agents import IntentRouter
from app.schemas import (
    ApprovalDecision,
    ApprovalResumeCommand,
    ApprovalStatus,
    ConversationTurn,
    DEFAULT_SPECIALIST_PERMISSIONS,
    GraphError,
    GraphState,
    Intent,
    IntentRouteResult,
    LongTermMemoryOperation,
    SpecialistInput,
    SpecialistKind,
    SpecialistPermissionPolicy,
    SpecialistResult,
    ThreadContext,
    TraceEvent,
    WorkflowStatus,
)
from app.services import (
    ContextManager,
    EduFlowStore,
    LLMLongTermMemoryExtractor,
    ModelProviderError,
)
from app.workflows.documentation_workflow import build_documentation_workflow
from app.workflows.family_workflow import build_family_workflow
from app.workflows.planning_workflow import build_planning_workflow
from app.workflows.policy_rag_graph import build_policy_rag_graph
from app.workflows.specialist import SpecialistWorkflowProtocol
from app.tools import build_default_tool_registry


GraphStateInput = Union[GraphState, Mapping[str, Any]]


class RouterProtocol(Protocol):
    def route(
        self,
        user_message: str,
        *,
        conversation_context: str = "",
    ) -> IntentRouteResult:
        ...


class ContextManagerProtocol(Protocol):
    def update_after_run(self, state: GraphState) -> Any:
        ...

    def build_model_context(
        self,
        context: ThreadContext,
        *,
        teacher_id: Optional[str] = None,
    ) -> str:
        ...


class LongTermMemoryExtractorProtocol(Protocol):
    def decide(
        self,
        *,
        turns: List[ConversationTurn],
        existing_memories: List[Dict[str, str]],
        teacher_id: Optional[str] = None,
        class_id: Optional[str] = None,
    ) -> List[LongTermMemoryOperation]:
        ...


class LongTermMemoryStoreProtocol(Protocol):
    def list_memories_for_owners(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        limit: int = 12,
    ) -> List[Dict[str, str]]:
        ...

    def apply_long_term_memory_operation(
        self,
        operation: LongTermMemoryOperation,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> Dict[str, str]:
        ...


class LearningRecordStoreProtocol(Protocol):
    def save_approved_learning_record(
        self,
        *,
        request_id: str,
        teacher_id: Optional[str],
        class_id: Optional[str],
        title: str,
        content: str,
    ) -> Dict[str, Any]:
        ...


def _state_from_input(state: GraphStateInput) -> GraphState:
    if isinstance(state, GraphState):
        return state
    return GraphState.model_validate(state)


def initialize(state: GraphStateInput) -> Dict[str, Any]:
    current_state = _state_from_input(state)
    thread_id = (
        current_state.thread_id
        or current_state.context.thread_id
        or current_state.session_id
    )
    context = current_state.context.model_copy(update={"thread_id": thread_id})
    return {
        "thread_id": thread_id,
        "context": context,
        "trace": [
            TraceEvent(
                step="initialize",
                message="Initialized EduFlow graph state.",
                metadata={"thread_id": thread_id},
            )
        ]
    }


def build_intent_router_node(
    router: RouterProtocol,
    context_manager: ContextManagerProtocol,
):
    def intent_router_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        try:
            route_result = router.route(
                current_state.user_message,
                conversation_context=context_manager.build_model_context(
                    current_state.context,
                    teacher_id=current_state.teacher_id,
                ),
            )
        except ModelProviderError as error:
            return {
                "intent": Intent.UNKNOWN,
                "needs_clarification": True,
                "clarification_question": (
                    "I could not reliably route that request. Could you say whether "
                    "you want an activity plan, learning record, policy answer, or "
                    "family message?"
                ),
                "workflow_status": WorkflowStatus.ROUTED,
                "errors": [
                    GraphError(
                        code=error.code.value,
                        message=(
                            "Intent routing was unavailable, so the request was "
                            "safely redirected to clarification."
                        ),
                        recoverable=True,
                    )
                ],
                "trace": [
                    TraceEvent(
                        step="intent_router",
                        message="Intent routing used the clarification fallback.",
                        metadata={
                            **error.safe_metadata(),
                            "fallback": "clarification",
                        },
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


def route_after_specialist(state: GraphStateInput) -> str:
    """Keep approval-gated work on an explicit main-graph path.

    Specialist subgraphs only declare that a controlled action needs approval.
    The main graph owns the transition into the common approval boundary, so
    Planning, Documentation, and future Family actions all pause consistently
    when interrupt/resume is introduced.
    """

    current_state = _state_from_input(state)
    if current_state.workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL:
        return "approval_gate"
    return "context_update"


def build_approval_gate_node(*, interrupts_enabled: bool):
    """Build the shared human-approval boundary.

    LangGraph interrupts need a checkpointer.  Graphs built without one retain
    the former pass-through behaviour so existing one-shot callers can still
    return a draft marked ``WAITING_FOR_APPROVAL``.  A checkpointed graph pauses
    here and exposes only safe approval metadata to the caller.
    """

    def approval_gate(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        payload = {
            "kind": "approval_required",
            "request_id": current_state.request_id,
            "risk_level": current_state.approval.risk_level.value,
            "reason": current_state.approval.reason,
            "allowed_decisions": ["approve", "reject"],
        }

        if interrupts_enabled:
            # Do not add side effects before interrupt(): this node is run again
            # from its start when a later Command(resume=...) continues it.
            resume_value = interrupt(payload)
            try:
                command = ApprovalResumeCommand.model_validate(resume_value)
            except ValidationError:
                return {
                    "workflow_status": WorkflowStatus.FAILED,
                    "errors": [
                        GraphError(
                            code="invalid_approval_resume",
                            message=(
                                "Approval response must include a valid request_id "
                                "and an approve or reject decision."
                            ),
                        )
                    ],
                    "trace": [
                        TraceEvent(
                            step="approval_gate",
                            message="Rejected an invalid approval response.",
                            metadata={"request_id": current_state.request_id},
                        )
                    ],
                }

            if command.request_id != current_state.request_id:
                return {
                    "workflow_status": WorkflowStatus.FAILED,
                    "errors": [
                        GraphError(
                            code="approval_request_mismatch",
                            message=(
                                "Approval response does not belong to the draft "
                                "currently waiting for review."
                            ),
                        )
                    ],
                    "trace": [
                        TraceEvent(
                            step="approval_gate",
                            message="Rejected an approval response for another request.",
                            metadata={"request_id": current_state.request_id},
                        )
                    ],
                }

            approved = command.decision is ApprovalDecision.APPROVE
            return {
                "workflow_status": (
                    WorkflowStatus.APPROVED_PENDING_SAVE
                    if approved
                    else WorkflowStatus.COMPLETED
                ),
                "approval": current_state.approval.model_copy(
                    update={
                        "status": (
                            ApprovalStatus.APPROVED
                            if approved
                            else ApprovalStatus.REJECTED
                        )
                    }
                ),
                "trace": [
                    TraceEvent(
                        step="approval_gate",
                        message=(
                            "Teacher approved draft; final save is pending."
                            if approved
                            else "Teacher rejected draft; no record was saved."
                        ),
                        metadata={
                            "request_id": current_state.request_id,
                            "decision": command.decision.value,
                        },
                    )
                ],
            }

        return {
            "trace": [
                TraceEvent(
                    step="approval_gate",
                    message="Draft is waiting for a teacher approval decision.",
                    metadata={
                        "request_id": current_state.request_id,
                        "risk_level": current_state.approval.risk_level.value,
                        "interrupt_enabled": interrupts_enabled,
                    },
                )
            ]
        }

    return approval_gate


def route_after_approval_gate(state: GraphStateInput) -> str:
    """Only an approved, resumed request may enter the final-save node."""

    current_state = _state_from_input(state)
    if current_state.workflow_status is WorkflowStatus.APPROVED_PENDING_SAVE:
        return "save_approved_learning_record"
    return "context_update"


def build_approved_learning_record_save_node(store: LearningRecordStoreProtocol):
    """Save an approved draft once, using request_id as the idempotency key."""

    def save_approved_learning_record(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        if current_state.approval.status is not ApprovalStatus.APPROVED:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "errors": [
                    GraphError(
                        code="learning_record_save_not_approved",
                        message="An approved teacher decision is required before saving.",
                    )
                ],
            }
        if current_state.draft is None:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "errors": [
                    GraphError(
                        code="learning_record_save_missing_draft",
                        message="Cannot save an approved learning record without a draft.",
                    )
                ],
            }

        saved = store.save_approved_learning_record(
            request_id=current_state.request_id,
            teacher_id=current_state.teacher_id,
            class_id=current_state.class_id,
            title=current_state.draft.title or "Learning record",
            content=current_state.draft.content,
        )
        return {
            "workflow_status": WorkflowStatus.COMPLETED,
            "trace": [
                TraceEvent(
                    step="learning_record_save",
                    message="Saved approved learning record.",
                    metadata={
                        "record_id": saved["record_id"],
                        "request_id": current_state.request_id,
                        "created": saved["created"],
                    },
                )
            ],
        }

    return save_approved_learning_record


def build_specialist_workflow_node(
    workflow: SpecialistWorkflowProtocol,
    specialist: SpecialistKind,
    context_manager: ContextManagerProtocol,
):
    def specialist_workflow_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        specialist_input = SpecialistInput.from_graph_state(
            current_state,
            specialist=specialist,
            conversation_context=context_manager.build_model_context(
                current_state.context,
                teacher_id=current_state.teacher_id,
            ),
        )
        result = SpecialistResult.model_validate(
            workflow.invoke(specialist_input)
        )
        if result.specialist is not specialist:
            raise ValueError(
                f"{specialist.value} workflow returned a "
                f"{result.specialist.value} result"
            )
        return result.to_graph_update()

    return specialist_workflow_node


def policy_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "trace": [
            TraceEvent(
                step="policy_placeholder",
                message="Routed to policy QA workflow placeholder.",
            )
        ]
    }


def clarification_placeholder(state: GraphStateInput) -> Dict[str, Any]:
    return {
        "workflow_status": WorkflowStatus.COMPLETED,
        "trace": [
            TraceEvent(
                step="clarification_placeholder",
                message="Routed to clarification placeholder.",
            )
        ]
    }


def build_context_update_node(context_manager: ContextManagerProtocol):
    def context_update_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        context = context_manager.update_after_run(current_state)
        return {
            "context": context,
            "trace": [
                TraceEvent(
                    step="context_update",
                    message="Updated compressed thread context.",
                    metadata={
                        "thread_id": context.thread_id,
                        "recent_turns": len(context.recent_turns),
                        "open_tasks": len(context.memory.open_tasks),
                        "summary_chars": len(context.memory.compact_summary),
                    },
                )
            ],
        }

    return context_update_node


def build_long_memory_update_node(
    extractor: LongTermMemoryExtractorProtocol,
    store: LongTermMemoryStoreProtocol,
):
    """Ask the LLM to consolidate memory after every completed graph turn."""

    def long_memory_update_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        if current_state.teacher_id is None and current_state.class_id is None:
            return {
                "trace": [
                    TraceEvent(
                        step="long_memory_update",
                        message="Skipped long-term memory update without an owner.",
                        metadata={"applied_operations": 0},
                    )
                ]
            }

        try:
            existing_memories = store.list_memories_for_owners(
                teacher_id=current_state.teacher_id,
                class_id=current_state.class_id,
            )
            operations = extractor.decide(
                turns=current_state.context.recent_turns[-2:],
                existing_memories=existing_memories,
                teacher_id=current_state.teacher_id,
                class_id=current_state.class_id,
            )
            applied = [
                store.apply_long_term_memory_operation(
                    operation,
                    teacher_id=current_state.teacher_id,
                    class_id=current_state.class_id,
                )
                for operation in operations
            ]
        except (ModelProviderError, TypeError, ValueError) as error:
            return {
                "trace": [
                    TraceEvent(
                        step="long_memory_update",
                        message="Long-term memory update was skipped after an error.",
                        metadata={"applied_operations": 0, "error": str(error)},
                    )
                ]
            }

        return {
            "trace": [
                TraceEvent(
                    step="long_memory_update",
                    message="Applied long-term memory operations.",
                    metadata={
                        "applied_operations": len(applied),
                        "actions": [item["action"] for item in applied],
                        "memory_ids": [
                            item["memory_id"] for item in applied if "memory_id" in item
                        ],
                    },
                )
            ]
        }

    return long_memory_update_node


def build_main_graph(
    router: Optional[RouterProtocol] = None,
    planning_workflow: Optional[SpecialistWorkflowProtocol] = None,
    policy_workflow: Optional[SpecialistWorkflowProtocol] = None,
    documentation_workflow: Optional[SpecialistWorkflowProtocol] = None,
    family_workflow: Optional[SpecialistWorkflowProtocol] = None,
    specialist_permissions: Optional[
        Mapping[SpecialistKind, SpecialistPermissionPolicy]
    ] = None,
    context_manager: Optional[ContextManagerProtocol] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    long_memory_extractor: Optional[LongTermMemoryExtractorProtocol] = None,
    long_memory_store: Optional[LongTermMemoryStoreProtocol] = None,
    learning_record_store: Optional[LearningRecordStoreProtocol] = None,
):
    resolved_specialist_permissions = _resolve_specialist_permissions(
        specialist_permissions
    )
    resolved_long_memory_store = long_memory_store or _default_long_memory_store()
    resolved_learning_record_store = learning_record_store or resolved_long_memory_store
    resolved_long_memory_extractor = long_memory_extractor or LLMLongTermMemoryExtractor()
    resolved_router = router or IntentRouter()
    resolved_planning_workflow = planning_workflow or build_planning_workflow(
        registry=(
            build_default_tool_registry(resolved_long_memory_store)
            if isinstance(resolved_long_memory_store, EduFlowStore)
            else None
        ),
        permission=resolved_specialist_permissions[SpecialistKind.PLANNING],
    )
    resolved_context_manager = context_manager or ContextManager(
        long_term_memory_reader=resolved_long_memory_store
    )
    resolved_policy_workflow = policy_workflow or build_policy_rag_graph(
        permission=resolved_specialist_permissions[SpecialistKind.POLICY]
    )
    resolved_documentation_workflow = (
        documentation_workflow
        or build_documentation_workflow(
            permission=resolved_specialist_permissions[SpecialistKind.DOCUMENTATION]
        )
    )
    resolved_family_workflow = family_workflow or build_family_workflow(
        permission=resolved_specialist_permissions[SpecialistKind.FAMILY]
    )
    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node(
        "intent_router",
        build_intent_router_node(resolved_router, resolved_context_manager),
    )
    graph.add_node(
        "planning_react",
        build_specialist_workflow_node(
            resolved_planning_workflow,
            SpecialistKind.PLANNING,
            resolved_context_manager,
        ),
    )
    graph.add_node(
        "documentation",
        build_specialist_workflow_node(
            resolved_documentation_workflow,
            SpecialistKind.DOCUMENTATION,
            resolved_context_manager,
        ),
    )
    graph.add_node(
        "policy_rag",
        build_specialist_workflow_node(
            resolved_policy_workflow,
            SpecialistKind.POLICY,
            resolved_context_manager,
        ),
    )
    graph.add_node(
        "family",
        build_specialist_workflow_node(
            resolved_family_workflow,
            SpecialistKind.FAMILY,
            resolved_context_manager,
        ),
    )
    graph.add_node("clarification_placeholder", clarification_placeholder)
    graph.add_node(
        "approval_gate",
        build_approval_gate_node(interrupts_enabled=checkpointer is not None),
    )
    graph.add_node(
        "save_approved_learning_record",
        build_approved_learning_record_save_node(resolved_learning_record_store),
    )
    graph.add_node("context_update", build_context_update_node(resolved_context_manager))
    graph.add_node(
        "long_memory_update",
        build_long_memory_update_node(
            resolved_long_memory_extractor,
            resolved_long_memory_store,
        ),
    )
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "intent_router")
    graph.add_conditional_edges(
        "intent_router",
        route_by_intent,
        {
            "planning": "planning_react",
            "documentation": "documentation",
            "policy": "policy_rag",
            "family": "family",
            "clarification": "clarification_placeholder",
            "end": "context_update",
        },
    )
    for specialist_node in (
        "planning_react",
        "documentation",
        "policy_rag",
        "family",
    ):
        graph.add_conditional_edges(
            specialist_node,
            route_after_specialist,
            {
                "approval_gate": "approval_gate",
                "context_update": "context_update",
            },
        )
    graph.add_conditional_edges(
        "approval_gate",
        route_after_approval_gate,
        {
            "save_approved_learning_record": "save_approved_learning_record",
            "context_update": "context_update",
        },
    )
    graph.add_edge("save_approved_learning_record", "context_update")
    graph.add_edge("clarification_placeholder", "context_update")
    graph.add_edge("context_update", "long_memory_update")
    graph.add_edge("long_memory_update", END)
    return graph.compile(checkpointer=checkpointer)


def _default_long_memory_store() -> EduFlowStore:
    store = EduFlowStore()
    store.initialize()
    return store


def _resolve_specialist_permissions(
    overrides: Optional[Mapping[SpecialistKind, SpecialistPermissionPolicy]],
) -> Dict[SpecialistKind, SpecialistPermissionPolicy]:
    resolved = dict(DEFAULT_SPECIALIST_PERMISSIONS)
    resolved.update(overrides or {})
    for specialist, permission in resolved.items():
        permission.require_specialist(specialist)
    return resolved
