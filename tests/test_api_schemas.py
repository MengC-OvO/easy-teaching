import pytest
from pydantic import ValidationError

from app.schemas import (
    ApiErrorDetail,
    ApiErrorResponse,
    Approval,
    ApprovalDecision,
    ApprovalStatus,
    ApprovalSubmitRequest,
    ApprovalSubmitResponse,
    CancelRunResponse,
    Citation,
    Draft,
    DraftResponse,
    MessageAcceptedResponse,
    MessageCreateRequest,
    RiskLevel,
    RunStatus,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionStatus,
    StreamEvent,
    StreamEventType,
)


def test_session_contract_strips_scoped_identifiers() -> None:
    request = SessionCreateRequest(
        teacher_id="  teacher-001  ",
        class_id=" class-001 ",
    )
    response = SessionCreateResponse(
        session_id=" session-001 ",
        thread_id=" thread-001 ",
    )

    assert request.teacher_id == "teacher-001"
    assert request.class_id == "class-001"
    assert response.session_id == "session-001"
    assert response.thread_id == "thread-001"
    assert response.status is SessionStatus.ACTIVE


def test_message_contract_accepts_an_optional_idempotency_key() -> None:
    request = MessageCreateRequest(
        message="  Plan an outdoor activity.  ",
        request_id=" req-client-001 ",
    )
    accepted = MessageAcceptedResponse(
        session_id="session-001",
        request_id=request.request_id,
    )

    assert request.message == "Plan an outdoor activity."
    assert request.request_id == "req-client-001"
    assert accepted.status is RunStatus.ACCEPTED


@pytest.mark.parametrize("message", ["", "   ", "x" * 20_001])
def test_message_contract_rejects_invalid_messages(message: str) -> None:
    with pytest.raises(ValidationError):
        MessageCreateRequest(message=message)


def test_api_contract_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SessionCreateRequest(teacher_id="teacher-001", centre_id="unexpected")


def test_draft_response_reuses_graph_domain_models() -> None:
    response = DraftResponse(
        session_id="session-001",
        request_id="req-001",
        status="waiting_for_approval",
        draft=Draft(title="Outdoor plan", content="Draft content"),
        approval=Approval(
            status=ApprovalStatus.REQUIRED,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
            reason="Teacher review is required before saving.",
        ),
        citations=[Citation(source="eylf-v2", section="Outcome 4")],
    )

    assert response.status is RunStatus.WAITING_FOR_APPROVAL
    assert response.draft.is_draft is True
    assert response.approval.status is ApprovalStatus.REQUIRED
    assert response.citations[0].source == "eylf-v2"


def test_approval_submit_request_reuses_resume_validation() -> None:
    request = ApprovalSubmitRequest(
        request_id=" req-approval-001 ",
        decision="approve",
    )

    assert request.request_id == "req-approval-001"
    assert request.decision is ApprovalDecision.APPROVE

    with pytest.raises(ValidationError):
        ApprovalSubmitRequest(request_id="req-approval-001", decision="save")


def test_approval_and_cancel_responses_use_run_lifecycle_statuses() -> None:
    approval = ApprovalSubmitResponse(
        session_id="session-001",
        request_id="req-approval-001",
        decision="approve",
        status="running",
    )
    cancelled = CancelRunResponse(
        session_id="session-001",
        request_id="req-cancel-001",
    )

    assert approval.decision is ApprovalDecision.APPROVE
    assert approval.status is RunStatus.RUNNING
    assert cancelled.status is RunStatus.CANCELLED


def test_stream_event_has_stable_name_order_and_payload() -> None:
    event = StreamEvent(
        event_id="event-001",
        event="approval_required",
        sequence=4,
        session_id="session-001",
        request_id="req-001",
        data={"reason": "Teacher review required"},
    )

    assert event.event is StreamEventType.APPROVAL_REQUIRED
    assert event.sequence == 4
    assert event.data == {"reason": "Teacher review required"}


def test_stream_event_rejects_negative_sequence() -> None:
    with pytest.raises(ValidationError):
        StreamEvent(
            event_id="event-001",
            event="trace",
            sequence=-1,
            session_id="session-001",
        )


def test_api_error_contract_carries_safe_structured_details() -> None:
    response = ApiErrorResponse(
        request_id="req-001",
        error=ApiErrorDetail(
            code="approval_conflict",
            message="The request is not waiting for approval.",
            recoverable=True,
            details={"current_status": "completed"},
        ),
    )

    assert response.error.code == "approval_conflict"
    assert response.error.recoverable is True
    assert response.error.details["current_status"] == "completed"
