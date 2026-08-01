import pytest
from pydantic import ValidationError

from evals import (
    EvalCase,
    EvalCategory,
    EvalCheck,
    EvalResult,
    EvalTokenUsage,
    MemoryExpectation,
    RoutingExpectation,
    SafetyExpectation,
    SafetyOutcome,
    ToolExpectation,
    TrajectoryExpectation,
)
from app.schemas import Intent


def test_routing_eval_case_has_typed_input_and_expectation() -> None:
    case = EvalCase(
        id="route-001",
        category="routing",
        input={"message": "  Plan an outdoor activity.  "},
        expected={
            "intent": "activity_planning",
            "needs_clarification": False,
        },
        tags=["smoke", "week4"],
    )

    assert case.category is EvalCategory.ROUTING
    assert case.input.message == "Plan an outdoor activity."
    assert isinstance(case.expected, RoutingExpectation)
    assert case.expected.intent is Intent.ACTIVITY_PLANNING


def test_eval_case_rejects_an_expectation_for_another_category() -> None:
    with pytest.raises(ValidationError, match="RoutingExpectation"):
        EvalCase(
            id="route-wrong-contract",
            category="routing",
            input={"message": "Plan an activity."},
            expected={"required_tool_names": ["get_class_profile"]},
        )


def test_tool_expectation_validates_required_arguments_and_denied_tools() -> None:
    expectation = ToolExpectation(
        required_tool_names=["get_class_profile"],
        forbidden_tool_names=["save_draft"],
        required_args={"get_class_profile": {"class_id": "kangaroo-room"}},
    )

    assert expectation.required_args["get_class_profile"] == {
        "class_id": "kangaroo-room"
    }

    with pytest.raises(ValidationError, match="non-required tools"):
        ToolExpectation(
            required_tool_names=["get_class_profile"],
            required_args={"save_draft": {}},
        )


def test_trajectory_contract_rejects_contradictory_steps() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        TrajectoryExpectation(
            required_steps=["intent_router"],
            forbidden_steps=["intent_router"],
        )


def test_memory_and_safety_expectations_select_explicit_paths() -> None:
    memory = MemoryExpectation(
        target="recall_tool",
        should_succeed=False,
        expected_error_code="permission_denied",
    )
    safety = SafetyExpectation(
        outcome="redact",
        forbidden_output_fragments=["parent@example.com"],
    )

    assert memory.target.value == "recall_tool"
    assert safety.outcome is SafetyOutcome.REDACT


def test_eval_contract_rejects_unknown_fixture_fields() -> None:
    with pytest.raises(ValidationError):
        EvalCase(
            id="route-extra-field",
            category="routing",
            input={"message": "Plan an activity.", "centre_id": "unexpected"},
            expected={"intent": "activity_planning"},
        )


def test_eval_result_normalizes_checks_timing_usage_and_cost() -> None:
    result = EvalResult(
        case_id="route-001",
        category="routing",
        passed=True,
        score=1.0,
        checks=[
            EvalCheck(
                name="intent_matches",
                passed=True,
                expected="activity_planning",
                actual="activity_planning",
            )
        ],
        latency_ms=18.5,
        token_usage=EvalTokenUsage(
            model_calls=1,
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
        estimated_cost_usd=0.0001,
    )

    assert result.passed is True
    assert result.checks[0].name == "intent_matches"
    assert result.token_usage.total_tokens == 28


def test_token_usage_rejects_an_impossible_total() -> None:
    with pytest.raises(ValidationError, match="total_tokens"):
        EvalTokenUsage(
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=10,
        )
