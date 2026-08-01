import sqlite3

from langgraph.types import Command

from app.schemas import (
    Approval,
    ApprovalStatus,
    Draft,
    GraphState,
    Intent,
    IntentRouteResult,
    LongTermMemoryAction,
    LongTermMemoryCandidate,
    LongTermMemoryOperation,
    LongTermMemoryScope,
    LongTermMemoryType,
    MemoryRetrievalMode,
    PolicyRAGResult,
    PolicyRAGStatus,
    RiskLevel,
    RetrievalResult,
    RetrievalStats,
    SpecialistInput,
    SpecialistKind,
    SpecialistResult,
    TraceEvent,
    ConversationMemory,
    ThreadContext,
    WorkflowStatus,
)
from app.services import ContextManager, EduFlowStore, ModelTimeoutError
from app.workflows import (
    build_main_graph,
    build_policy_rag_graph,
    build_sqlite_checkpointer,
    checkpoint_config,
)


class StubRouter:
    def __init__(self, result: IntentRouteResult) -> None:
        self.result = result
        self.user_message = None

    def route(self, user_message: str, *, conversation_context: str = "") -> IntentRouteResult:
        self.user_message = user_message
        return self.result


class RecordingRouter(StubRouter):
    def __init__(self, result: IntentRouteResult) -> None:
        super().__init__(result)
        self.contexts = []

    def route(self, user_message: str, *, conversation_context: str = "") -> IntentRouteResult:
        self.contexts.append(conversation_context)
        return super().route(user_message, conversation_context=conversation_context)


class SequencedMemoryExtractor:
    def __init__(self, decisions) -> None:
        self.decisions = iter(decisions)

    def decide(self, *, turns, existing_memories, teacher_id=None, class_id=None):
        return next(self.decisions)


class FailingRouter:
    def route(self, user_message: str, *, conversation_context: str = "") -> IntentRouteResult:
        raise ModelTimeoutError("router timed out")


class StubPlanningWorkflow:
    def __init__(self, result: SpecialistResult) -> None:
        self.result = result
        self.input_state = None

    def invoke(self, state: SpecialistInput):
        self.input_state = state
        return self.result


class StubPolicyWorkflow:
    def __init__(self, result: SpecialistResult) -> None:
        self.result = result
        self.input_state = None

    def invoke(self, state: SpecialistInput):
        self.input_state = state
        return self.result


class StubDocumentationWorkflow(StubPolicyWorkflow):
    pass


class ContextRecordingPolicyService:
    def __init__(self) -> None:
        self.contexts = []

    def answer(self, question: str, *, conversation_context: str = "") -> PolicyRAGResult:
        self.contexts.append(conversation_context)
        return PolicyRAGResult(
            status=PolicyRAGStatus.ANSWERED,
            question=question,
            answer="Policy answer draft.",
            retrieval=RetrievalResult(
                query=question,
                chunks=[],
                stats=RetrievalStats(
                    requested_top_k=5,
                    raw_result_count=0,
                    deduplicated_count=0,
                    returned_count=0,
                ),
            ),
        )


def test_main_graph_runs_intent_router_node() -> None:
    router = StubRouter(
        IntentRouteResult(
            intent=Intent.ACTIVITY_PLANNING,
            confidence=0.9,
            reason="The request asks for an activity plan.",
        )
    )
    planning_workflow = StubPlanningWorkflow(
        SpecialistResult(
            specialist=SpecialistKind.PLANNING,
            status=WorkflowStatus.COMPLETED,
            draft=Draft(
                title="Activity planning draft",
                content="Draft activity plan.",
            ),
            trace=[
                TraceEvent(
                    step="planning_react",
                    message="Activity planning ReAct workflow completed.",
                )
            ],
        )
    )
    graph = build_main_graph(router, planning_workflow=planning_workflow)
    initial_state = GraphState(
        request_id="req-graph-001",
        session_id="session-001",
        user_message="Plan an outdoor activity.",
    )

    result = graph.invoke(initial_state)
    final_state = GraphState.model_validate(result)

    assert final_state.workflow_status is WorkflowStatus.COMPLETED
    assert final_state.intent is Intent.ACTIVITY_PLANNING
    assert final_state.draft is not None
    assert final_state.draft.content == "Draft activity plan."
    assert router.user_message == "Plan an outdoor activity."
    assert planning_workflow.input_state.user_message == "Plan an outdoor activity."
    assert [event.step for event in final_state.trace] == [
        "initialize",
        "intent_router",
        "planning_react",
        "context_update",
        "long_memory_update",
    ]
    assert final_state.thread_id == "session-001"
    assert final_state.context.thread_id == "session-001"


def test_main_graph_passes_compact_context_to_react_workflow() -> None:
    planning_workflow = StubPlanningWorkflow(
        SpecialistResult(
            specialist=SpecialistKind.PLANNING,
            status=WorkflowStatus.COMPLETED,
            draft=Draft(content="Shortened draft."),
        )
    )
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.ACTIVITY_PLANNING,
                confidence=0.9,
                reason="Follow-up to an activity plan.",
            )
        ),
        planning_workflow=planning_workflow,
    )

    graph.invoke(
        GraphState(
            request_id="req-context-injection",
            session_id="session-context-injection",
            user_message="Make it shorter.",
            context=ThreadContext(
                memory=ConversationMemory(
                    compact_summary="Teacher is revising an outdoor activity plan."
                )
            ),
        )
    )

    assert "outdoor activity plan" in planning_workflow.input_state.conversation_context


def test_main_graph_recalls_a_memory_written_by_the_previous_turn(tmp_path) -> None:
    store = EduFlowStore(database_url=f"sqlite:///{tmp_path / 'eduflow.sqlite3'}")
    store.initialize()
    preference = LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id="teacher-001",
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content="Prefers concise activity-plan steps.",
        reason="The teacher explicitly requested short steps.",
        retrieval_mode=MemoryRetrievalMode.PROFILE,
        importance=4,
    )
    extractor = SequencedMemoryExtractor(
        [
            [
                LongTermMemoryOperation(
                    action=LongTermMemoryAction.INSERT,
                    candidate=preference,
                    reason="A durable preference was explicitly stated.",
                )
            ],
            [],
        ]
    )
    router = RecordingRouter(
        IntentRouteResult(
            intent=Intent.LEARNING_RECORD,
            confidence=0.9,
            reason="The request asks for a learning record.",
        )
    )
    graph = build_main_graph(
        router,
        long_memory_extractor=extractor,
        long_memory_store=store,
    )

    first_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-memory-first",
                session_id="session-memory",
                teacher_id="teacher-001",
                user_message="For future plans, keep the steps concise.",
            )
        )
    )
    graph.invoke(
        GraphState(
            request_id="req-memory-second",
            session_id="session-memory",
            teacher_id="teacher-001",
            user_message="Write a learning story.",
            context=first_state.context,
        )
    )

    assert "Prefers concise activity-plan steps." not in router.contexts[0]
    assert "Prefers concise activity-plan steps." in router.contexts[1]


def test_main_graph_passes_profile_memory_to_policy_rag_in_a_new_session(tmp_path) -> None:
    store = EduFlowStore(database_url=f"sqlite:///{tmp_path / 'eduflow.sqlite3'}")
    store.initialize()
    preference = LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id="teacher-001",
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content="Prefers concise policy-answer summaries.",
        reason="The teacher explicitly requested concise summaries.",
        retrieval_mode=MemoryRetrievalMode.PROFILE,
        importance=4,
    )
    extractor = SequencedMemoryExtractor(
        [
            [
                LongTermMemoryOperation(
                    action=LongTermMemoryAction.INSERT,
                    candidate=preference,
                    reason="A durable preference was explicitly stated.",
                )
            ],
            [],
        ]
    )
    router = StubRouter(
        IntentRouteResult(
            intent=Intent.POLICY_QA,
            confidence=0.9,
            reason="The request is a policy question.",
        )
    )
    context_manager = ContextManager(long_term_memory_reader=store)
    policy_service = ContextRecordingPolicyService()
    graph = build_main_graph(
        router,
        policy_workflow=build_policy_rag_graph(policy_service),
        context_manager=context_manager,
        long_memory_extractor=extractor,
        long_memory_store=store,
    )

    graph.invoke(
        GraphState(
            request_id="req-policy-memory-first",
            session_id="session-policy-memory-first",
            teacher_id="teacher-001",
            user_message="For future policy answers, keep summaries concise.",
        )
    )
    graph.invoke(
        GraphState(
            request_id="req-policy-memory-second",
            session_id="session-policy-memory-second",
            teacher_id="teacher-001",
            user_message="What does the EYLF say about play-based learning?",
        )
    )

    assert "Prefers concise policy-answer summaries." not in policy_service.contexts[0]
    assert "Prefers concise policy-answer summaries." in policy_service.contexts[1]


def test_main_graph_can_checkpoint_state_to_sqlite(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.LEARNING_RECORD,
                confidence=0.9,
                reason="The request asks for documentation.",
            )
        ),
        checkpointer=build_sqlite_checkpointer(checkpoint_path),
    )

    result = graph.invoke(
        GraphState(
            request_id="req-checkpoint",
            session_id="session-checkpoint",
            thread_id="thread-checkpoint",
            user_message="Write a learning story draft.",
        ),
        config=checkpoint_config("thread-checkpoint"),
    )
    final_state = GraphState.model_validate(result)

    assert final_state.thread_id == "thread-checkpoint"
    assert final_state.context.thread_id == "thread-checkpoint"
    assert checkpoint_path.exists()

    with sqlite3.connect(checkpoint_path) as connection:
        checkpoint_count = connection.execute("select count(*) from checkpoints").fetchone()[0]

    assert checkpoint_count > 0


def test_main_graph_interrupts_at_approval_gate_when_checkpointed(tmp_path) -> None:
    checkpoint_path = tmp_path / "approval-checkpoints.sqlite3"
    store = EduFlowStore(database_url=f"sqlite:///{tmp_path / 'eduflow.sqlite3'}")
    store.initialize()
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.LEARNING_RECORD,
                confidence=0.9,
                reason="The request asks for documentation.",
            )
        ),
        documentation_workflow=StubDocumentationWorkflow(
            SpecialistResult(
                specialist=SpecialistKind.DOCUMENTATION,
                status=WorkflowStatus.WAITING_FOR_APPROVAL,
                draft=Draft(title="Learning record draft", content="{}"),
                approval=Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason="Teacher review is required before saving.",
                ),
            )
        ),
        checkpointer=build_sqlite_checkpointer(checkpoint_path),
        long_memory_store=store,
        learning_record_store=store,
    )
    config = checkpoint_config("thread-approval-interrupt")

    events = list(
        graph.stream(
            GraphState(
                request_id="req-approval-interrupt",
                session_id="session-approval-interrupt",
                thread_id="thread-approval-interrupt",
                user_message="Write a learning story draft.",
            ),
            config=config,
        )
    )

    interrupt_event = next(event for event in events if "__interrupt__" in event)
    approval_interrupt = interrupt_event["__interrupt__"][0]
    assert approval_interrupt.value == {
        "kind": "approval_required",
        "request_id": "req-approval-interrupt",
        "risk_level": "L2_controlled_write",
        "reason": "Teacher review is required before saving.",
        "allowed_decisions": ["approve", "reject"],
    }
    paused_state = GraphState.model_validate(graph.get_state(config).values)
    assert paused_state.workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert paused_state.approval.status is ApprovalStatus.REQUIRED
    assert paused_state.trace[-1].step != "context_update"

    resumed_state = GraphState.model_validate(
        graph.invoke(
            Command(
                resume={
                    "request_id": "req-approval-interrupt",
                    "decision": "approve",
                }
            ),
            config=config,
        )
    )
    assert resumed_state.workflow_status is WorkflowStatus.COMPLETED
    assert resumed_state.approval.status is ApprovalStatus.APPROVED
    assert resumed_state.trace[-4].metadata == {
        "request_id": "req-approval-interrupt",
        "decision": "approve",
    }
    save_trace = resumed_state.trace[-3]
    assert save_trace.step == "learning_record_save"
    assert save_trace.metadata["request_id"] == "req-approval-interrupt"
    assert save_trace.metadata["created"] is True
    assert [event.step for event in resumed_state.trace[-2:]] == [
        "context_update",
        "long_memory_update",
    ]
    saved = store.get_learning_record_by_request_id("req-approval-interrupt")
    assert saved is not None
    assert saved["title"] == "Learning record draft"


def test_main_graph_rejects_resume_for_a_different_request(tmp_path) -> None:
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.LEARNING_RECORD,
                confidence=0.9,
                reason="The request asks for documentation.",
            )
        ),
        documentation_workflow=StubDocumentationWorkflow(
            SpecialistResult(
                specialist=SpecialistKind.DOCUMENTATION,
                status=WorkflowStatus.WAITING_FOR_APPROVAL,
                approval=Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                ),
            )
        ),
        checkpointer=build_sqlite_checkpointer(tmp_path / "mismatch-checkpoints.sqlite3"),
    )
    config = checkpoint_config("thread-approval-mismatch")
    list(
        graph.stream(
            GraphState(
                request_id="req-waiting",
                session_id="session-waiting",
                thread_id="thread-approval-mismatch",
                user_message="Write a learning story draft.",
            ),
            config=config,
        )
    )

    final_state = GraphState.model_validate(
        graph.invoke(
            Command(
                resume={
                    "request_id": "req-other",
                    "decision": "approve",
                }
            ),
            config=config,
        )
    )

    assert final_state.workflow_status is WorkflowStatus.FAILED
    assert final_state.approval.status is ApprovalStatus.REQUIRED
    assert final_state.errors[-1].code == "approval_request_mismatch"


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


def test_main_graph_falls_back_to_clarification_when_router_fails() -> None:
    graph = build_main_graph(FailingRouter())

    result = graph.invoke(
        {
            "request_id": "req-graph-003",
            "session_id": "session-003",
            "user_message": "Plan an activity.",
        }
    )
    final_state = GraphState.model_validate(result)

    assert final_state.workflow_status is WorkflowStatus.COMPLETED
    assert final_state.intent is Intent.UNKNOWN
    assert final_state.needs_clarification is True
    assert "activity plan" in final_state.clarification_question
    assert final_state.errors[0].code == "timeout"
    assert final_state.errors[0].recoverable is True
    assert [event.step for event in final_state.trace[-4:]] == [
        "intent_router",
        "clarification_placeholder",
        "context_update",
        "long_memory_update",
    ]


def test_main_graph_routes_learning_record_to_documentation_specialist() -> None:
    documentation_workflow = StubDocumentationWorkflow(
        SpecialistResult(
            specialist=SpecialistKind.DOCUMENTATION,
            status=WorkflowStatus.WAITING_FOR_APPROVAL,
            draft=Draft(
                title="Learning record draft",
                content='{"is_draft": true}',
            ),
            approval=Approval(
                status=ApprovalStatus.REQUIRED,
                risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                reason="Teacher review is required before saving.",
            ),
            trace=[
                TraceEvent(
                    step="documentation_draft",
                    message="Generated a draft for teacher review.",
                )
            ],
        )
    )
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.LEARNING_RECORD,
                confidence=0.9,
                reason="The request asks for a learning record.",
            )
        ),
        documentation_workflow=documentation_workflow,
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

    assert final_state.workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert final_state.draft is not None
    assert final_state.draft.title == "Learning record draft"
    assert final_state.draft.is_draft is True
    assert documentation_workflow.input_state.specialist is SpecialistKind.DOCUMENTATION
    assert [event.step for event in final_state.trace[-4:]] == [
        "documentation_draft",
        "approval_gate",
        "context_update",
        "long_memory_update",
    ]
    assert final_state.trace[-3].metadata == {
        "request_id": "req-doc",
        "risk_level": "L2_controlled_write",
        "interrupt_enabled": False,
    }


def test_main_graph_maps_planning_approval_required_to_workflow_state() -> None:
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.ACTIVITY_PLANNING,
                confidence=0.9,
                reason="The request asks for planning.",
            )
        ),
        planning_workflow=StubPlanningWorkflow(
            SpecialistResult(
                specialist=SpecialistKind.PLANNING,
                status=WorkflowStatus.WAITING_FOR_APPROVAL,
                approval=Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason="A controlled write tool requires teacher approval.",
                ),
                trace=[
                    TraceEvent(
                        step="planning_react",
                        message="Activity planning ReAct workflow completed.",
                        metadata={
                            "stop_reason": "approval_required",
                            "current_step": 1,
                            "observations": [
                                {
                                    "tool_name": "save_draft",
                                    "success": False,
                                    "error_code": "permission_denied",
                                }
                            ],
                        },
                    )
                ],
            )
        ),
    )

    final_state = GraphState.model_validate(
        graph.invoke(
            {
                "request_id": "req-approval",
                "session_id": "session-approval",
                "user_message": "Plan and save an activity draft.",
            }
        )
    )

    assert final_state.workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert final_state.approval.status is ApprovalStatus.REQUIRED
    planning_trace = final_state.trace[-4]
    assert planning_trace.step == "planning_react"
    assert planning_trace.metadata["stop_reason"] == "approval_required"
    assert planning_trace.metadata["observations"] == [
        {
            "tool_name": "save_draft",
            "success": False,
            "error_code": "permission_denied",
        }
    ]
    assert final_state.trace[-3].step == "approval_gate"


def test_main_graph_routes_policy_qa_to_policy_placeholder() -> None:
    policy_result = SpecialistResult(
        specialist=SpecialistKind.POLICY,
        status=WorkflowStatus.COMPLETED,
        draft=Draft(content="Policy answer draft."),
    )
    policy_workflow = StubPolicyWorkflow(policy_result)
    graph = build_main_graph(
        StubRouter(
            IntentRouteResult(
                intent=Intent.POLICY_QA,
                confidence=0.88,
                reason="The request asks about policy.",
            )
        ),
        policy_workflow=policy_workflow,
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

    assert final_state.workflow_status is WorkflowStatus.COMPLETED
    assert policy_workflow.input_state.user_message == "What does NQS QA1 require?"


def test_main_graph_routes_family_communication_to_family_workflow() -> None:
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

    assert final_state.workflow_status is WorkflowStatus.COMPLETED
    assert final_state.draft is not None
    assert final_state.draft.title == "Family communication draft"
    assert final_state.draft.is_draft is True
    assert final_state.safety_flags[-1].code == "draft_only"
    assert final_state.trace[-3].step == "family_draft_skeleton"
    assert final_state.trace[-2].step == "context_update"
    assert final_state.trace[-1].step == "long_memory_update"


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
    assert final_state.trace[-3].step == "clarification_placeholder"
    assert final_state.trace[-2].step == "context_update"
    assert final_state.trace[-1].step == "long_memory_update"
