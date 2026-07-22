from pydantic import ValidationError

from app.schemas import (
    ApprovalStatus,
    Citation,
    ContextBudget,
    ConversationMemory,
    ConversationRole,
    ConversationTurn,
    Draft,
    GraphState,
    Intent,
    RiskLevel,
    SafetyFlag,
    ThreadContext,
    TraceEvent,
    WorkflowStatus,
)


def test_graph_state_has_safe_defaults() -> None:
    state = GraphState(
        request_id="req-001",
        session_id="session-001",
        user_message="Plan a sensory activity for preschool children.",
    )

    assert state.intent is Intent.UNKNOWN
    assert state.needs_clarification is False
    assert state.clarification_question is None
    assert state.workflow_status is WorkflowStatus.CREATED
    assert state.draft is None
    assert state.citations == []
    assert state.approval.status is ApprovalStatus.NOT_REQUIRED
    assert state.approval.risk_level is RiskLevel.L1_DRAFT
    assert state.trace == []
    assert state.errors == []
    assert state.safety_flags == []
    assert state.thread_id is None
    assert state.context.recent_turns == []
    assert state.context.memory.conversation_goal is None
    assert state.context.memory.open_tasks == []


def test_graph_state_can_hold_workflow_outputs() -> None:
    state = GraphState(
        request_id="req-002",
        session_id="session-001",
        user_message="Write a draft family update.",
        intent=Intent.FAMILY_COMMUNICATION,
        workflow_status=WorkflowStatus.DRAFTING,
        draft=Draft(title="Family update", content="Draft message"),
        citations=[Citation(source="synthetic-policy", section="communication")],
        trace=[TraceEvent(step="router", message="Routed to family communication")],
        safety_flags=[
            SafetyFlag(
                code="draft_only",
                message="Family communication must remain a draft.",
                risk_level=RiskLevel.L1_DRAFT,
            )
        ],
    )

    assert state.intent is Intent.FAMILY_COMMUNICATION
    assert state.draft is not None
    assert state.draft.is_draft is True
    assert state.citations[0].source == "synthetic-policy"
    assert state.trace[0].step == "router"
    assert state.safety_flags[0].risk_level is RiskLevel.L1_DRAFT


def test_graph_state_requires_request_session_and_message() -> None:
    try:
        GraphState()  # type: ignore[call-arg]
    except ValidationError as error:
        missing_fields = {err["loc"][0] for err in error.errors()}
    else:
        raise AssertionError("GraphState should require core identifiers")

    assert missing_fields == {"request_id", "session_id", "user_message"}


def test_thread_context_applies_context_budget() -> None:
    context = ThreadContext(
        thread_id="thread-001",
        recent_turns=[
            ConversationTurn(role=ConversationRole.USER, content=f"user {index}")
            for index in range(5)
        ],
        memory=ConversationMemory(
            compact_summary="x" * 240,
            important_requirements=["a", "b", "a", "c"],
        ),
        tool_trace_summary=[
            TraceEvent(step=f"tool_{index}", message="Called tool.")
            for index in range(5)
        ],
        budget=ContextBudget(
            max_recent_turns=2,
            max_trace_events=3,
            max_memory_summary_chars=200,
            max_memory_items=2,
        ),
    )

    assert [turn.content for turn in context.recent_turns] == ["user 3", "user 4"]
    assert context.memory.compact_summary == "x" * 200
    assert context.memory.important_requirements == ["b", "c"]
    assert [event.step for event in context.tool_trace_summary] == [
        "tool_2",
        "tool_3",
        "tool_4",
    ]
