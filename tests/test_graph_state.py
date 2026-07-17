from pydantic import ValidationError

from app.schemas import (
    ApprovalStatus,
    Citation,
    Draft,
    GraphState,
    Intent,
    RiskLevel,
    SafetyFlag,
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
