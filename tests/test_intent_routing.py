import pytest
from pydantic import ValidationError

from app.schemas import Intent, IntentRouteResult


def test_intent_route_result_accepts_confident_route() -> None:
    result = IntentRouteResult(
        intent=Intent.ACTIVITY_PLANNING,
        confidence=0.91,
        reason="The request asks for an activity plan.",
    )

    assert result.is_confident_route is True
    assert result.needs_clarification is False
    assert result.clarification_question is None


def test_intent_route_result_accepts_clarification_route() -> None:
    result = IntentRouteResult(
        intent=Intent.UNKNOWN,
        confidence=0.35,
        needs_clarification=True,
        clarification_question="Do you want an activity plan or a family message draft?",
        reason="The request is ambiguous between two workflows.",
    )

    assert result.is_confident_route is False
    assert result.needs_clarification is True


def test_intent_route_result_requires_clarification_question_when_needed() -> None:
    with pytest.raises(ValidationError):
        IntentRouteResult(
            intent=Intent.UNKNOWN,
            confidence=0.2,
            needs_clarification=True,
            reason="The request is ambiguous.",
        )


def test_intent_route_result_rejects_clarification_question_for_confident_route() -> None:
    with pytest.raises(ValidationError):
        IntentRouteResult(
            intent=Intent.POLICY_QA,
            confidence=0.82,
            clarification_question="Which policy do you mean?",
            reason="The request asks about a policy.",
        )


def test_intent_route_result_validates_confidence_range() -> None:
    with pytest.raises(ValidationError):
        IntentRouteResult(
            intent=Intent.FAMILY_COMMUNICATION,
            confidence=1.5,
            reason="The request asks for a family message.",
        )
