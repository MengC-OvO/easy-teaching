import pytest

from evals import (
    EvalTokenUsage,
    MemoryActual,
    ObservedToolCall,
    RagActual,
    RoutingActual,
    SafetyActual,
    ToolActual,
    TrajectoryActual,
    evaluate_case,
    load_eval_cases,
)


def _case(case_id):
    return next(case for case in load_eval_cases() if case.id == case_id)


def test_routing_evaluator_compares_intent_and_clarification() -> None:
    result = evaluate_case(
        _case("routing-activity-planning"),
        RoutingActual(intent="activity_planning", needs_clarification=False),
        latency_ms=12.5,
        token_usage=EvalTokenUsage(model_calls=1),
    )

    assert result.passed is True
    assert result.score == 1.0
    assert [check.name for check in result.checks] == [
        "intent_matches",
        "clarification_matches",
    ]


def test_tool_evaluator_checks_names_arguments_denials_and_order() -> None:
    result = evaluate_case(
        _case("tool-read-class-profile"),
        ToolActual(
            calls=[
                ObservedToolCall(
                    tool_name="load_skill",
                    tool_args={"skill_name": "activity_planning"},
                ),
                ObservedToolCall(
                    tool_name="get_class_profile",
                    tool_args={"class_id": "kangaroo-room", "unused": "allowed"},
                ),
            ]
        ),
    )

    assert result.passed is True
    assert result.score == 1.0


def test_rag_evaluator_scores_status_source_and_citation_count() -> None:
    result = evaluate_case(
        _case("rag-play-based-learning"),
        RagActual(
            status="answered",
            sources=["eylf-v2"],
            citation_count=3,
        ),
    )

    assert result.passed is True
    assert result.score == 1.0


def test_memory_evaluator_detects_cross_owner_content() -> None:
    result = evaluate_case(
        _case("memory-no-cross-teacher-leak"),
        MemoryActual(
            success=True,
            output="Previously used water-play activities outdoors and indoors.",
        ),
    )

    assert result.passed is False
    assert result.score == pytest.approx(2 / 3)
    assert next(
        check for check in result.checks if check.name == "forbidden_memory_1"
    ).passed is False


def test_safety_evaluator_checks_redaction_output() -> None:
    result = evaluate_case(
        _case("safety-redact-child-name"),
        SafetyActual(
            outcome="redact",
            output="Child named [PERSON_NAME_1] built a tall block tower.",
        ),
    )

    assert result.passed is True


def test_trajectory_evaluator_allows_extra_steps_but_preserves_required_order() -> None:
    result = evaluate_case(
        _case("trajectory-policy-route"),
        TrajectoryActual(
            steps=[
                "initialize",
                "extra_safe_trace",
                "intent_router",
                "policy_rag",
                "context_update",
                "long_memory_update",
            ]
        ),
    )

    assert result.passed is True


def test_evaluator_rejects_actual_data_for_another_category() -> None:
    with pytest.raises(TypeError, match="RoutingActual"):
        evaluate_case(
            _case("routing-policy-question"),
            RagActual(status="answered", sources=["eylf-v2"], citation_count=1),
        )
