import pytest
from pydantic import ValidationError

from app.schemas import ApprovalDecision, ApprovalResumeCommand


def test_approval_resume_command_accepts_an_approve_decision() -> None:
    command = ApprovalResumeCommand(
        request_id="  documentation-request-123  ",
        decision="approve",
    )

    assert command.request_id == "documentation-request-123"
    assert command.decision is ApprovalDecision.APPROVE


def test_approval_resume_command_accepts_a_reject_decision() -> None:
    command = ApprovalResumeCommand(
        request_id="planning-request-456",
        decision=ApprovalDecision.REJECT,
    )

    assert command.decision is ApprovalDecision.REJECT


@pytest.mark.parametrize("request_id", ["", "x" * 129])
def test_approval_resume_command_rejects_invalid_request_id(request_id: str) -> None:
    with pytest.raises(ValidationError):
        ApprovalResumeCommand(request_id=request_id, decision="approve")


def test_approval_resume_command_rejects_unknown_teacher_action() -> None:
    with pytest.raises(ValidationError):
        ApprovalResumeCommand(request_id="documentation-request-123", decision="save")
