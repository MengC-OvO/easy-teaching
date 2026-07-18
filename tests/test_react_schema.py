import pytest
from pydantic import ValidationError

from app.schemas import (
    Observation,
    ReActAction,
    ReActDecision,
    ReActState,
    StopReason,
    ToolCall,
)


def test_react_decision_accepts_tool_call_action() -> None:
    decision = ReActDecision(
        action=ReActAction.CALL_TOOL,
        tool_call=ToolCall(
            tool_name="get_class_profile",
            tool_args={"class_id": "kangaroo-room"},
        ),
        reason="Need class context before drafting.",
    )

    assert decision.action is ReActAction.CALL_TOOL
    assert decision.tool_call is not None
    assert decision.tool_call.tool_name == "get_class_profile"


def test_react_decision_accepts_final_answer_action() -> None:
    decision = ReActDecision(
        action=ReActAction.FINAL_ANSWER,
        final_answer="Use outdoor sensory exploration with synthetic materials.",
        reason="Enough information is available.",
    )

    assert decision.action is ReActAction.FINAL_ANSWER
    assert decision.final_answer.startswith("Use outdoor")


def test_react_decision_requires_tool_call_for_tool_action() -> None:
    with pytest.raises(ValidationError):
        ReActDecision(
            action=ReActAction.CALL_TOOL,
            reason="Need a tool.",
        )


def test_react_decision_requires_final_answer_for_final_action() -> None:
    with pytest.raises(ValidationError):
        ReActDecision(
            action=ReActAction.FINAL_ANSWER,
            reason="Ready to answer.",
        )


def test_react_state_tracks_step_budget_and_observations() -> None:
    state = ReActState(
        user_message="Plan an outdoor sensory activity.",
        max_steps=2,
        current_step=1,
        observations=[
            Observation(
                tool_name="get_class_profile",
                success=True,
                data={"class_id": "kangaroo-room"},
            )
        ],
    )

    assert state.has_steps_remaining is True
    assert state.should_stop is False
    assert state.observations[0].tool_name == "get_class_profile"


def test_react_state_stop_reason_marks_stopped() -> None:
    state = ReActState(
        user_message="Plan an outdoor sensory activity.",
        stop_reason=StopReason.COMPLETED,
    )

    assert state.should_stop is True
