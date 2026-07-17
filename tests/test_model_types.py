import pytest
from pydantic import BaseModel, ValidationError

from app.services import ModelMessage, ModelRequest, ModelResponse, ModelRole


class RouteResult(BaseModel):
    intent: str
    confidence: float


def test_model_request_accepts_messages_and_response_model() -> None:
    request = ModelRequest(
        messages=[
            ModelMessage(role=ModelRole.SYSTEM, content="Classify intent."),
            ModelMessage(role=ModelRole.USER, content="Plan a sensory activity."),
        ],
        response_model=RouteResult,
    )

    assert request.messages[0].role is ModelRole.SYSTEM
    assert request.response_model is RouteResult
    assert request.temperature == 0.0


def test_model_request_requires_at_least_one_message() -> None:
    with pytest.raises(ValidationError):
        ModelRequest(messages=[])


def test_model_request_validates_temperature_range() -> None:
    with pytest.raises(ValidationError):
        ModelRequest(
            messages=[ModelMessage(role=ModelRole.USER, content="hello")],
            temperature=3.0,
        )


def test_model_response_can_hold_structured_output() -> None:
    structured = RouteResult(intent="activity_planning", confidence=0.9)
    response = ModelResponse(
        content='{"intent": "activity_planning", "confidence": 0.9}',
        model="gemini-2.5-flash",
        finish_reason="stop",
        structured=structured,
    )

    assert response.structured == structured
    assert response.usage.total_tokens is None
