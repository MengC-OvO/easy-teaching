"""Deterministic category scorers for the offline evaluation harness."""

from typing import Any, Dict, Iterable, List, Optional, Type

from evals.schemas import (
    EvalActual,
    EvalCase,
    EvalCategory,
    EvalCheck,
    EvalResult,
    EvalTokenUsage,
    MemoryActual,
    MemoryExpectation,
    RagActual,
    RagExpectation,
    RoutingActual,
    RoutingExpectation,
    SafetyActual,
    SafetyExpectation,
    ToolActual,
    ToolExpectation,
    TrajectoryActual,
    TrajectoryExpectation,
)


_ACTUAL_TYPES: Dict[EvalCategory, Type[EvalActual]] = {
    EvalCategory.ROUTING: RoutingActual,
    EvalCategory.TOOL: ToolActual,
    EvalCategory.RAG: RagActual,
    EvalCategory.MEMORY: MemoryActual,
    EvalCategory.SAFETY: SafetyActual,
    EvalCategory.TRAJECTORY: TrajectoryActual,
}


def evaluate_case(
    case: EvalCase,
    actual: EvalActual,
    *,
    latency_ms: float = 0.0,
    token_usage: Optional[EvalTokenUsage] = None,
    estimated_cost_usd: float = 0.0,
    error_code: Optional[str] = None,
) -> EvalResult:
    """Compare one typed actual outcome with its fixture expectation."""
    actual_type = _ACTUAL_TYPES[case.category]
    if not isinstance(actual, actual_type):
        raise TypeError(
            f"{case.category.value} cases require {actual_type.__name__}, "
            f"got {type(actual).__name__}"
        )

    checks = _checks_for(case, actual)
    passed_count = sum(check.passed for check in checks)
    score = passed_count / len(checks) if checks else 0.0
    return EvalResult(
        case_id=case.id,
        category=case.category,
        passed=bool(checks) and passed_count == len(checks),
        score=score,
        checks=checks,
        latency_ms=latency_ms,
        token_usage=token_usage or EvalTokenUsage(),
        estimated_cost_usd=estimated_cost_usd,
        error_code=error_code,
    )


def _checks_for(case: EvalCase, actual: EvalActual) -> List[EvalCheck]:
    if case.category is EvalCategory.ROUTING:
        return _routing_checks(case.expected, actual)
    if case.category is EvalCategory.TOOL:
        return _tool_checks(case.expected, actual)
    if case.category is EvalCategory.RAG:
        return _rag_checks(case.expected, actual)
    if case.category is EvalCategory.MEMORY:
        return _memory_checks(case.expected, actual)
    if case.category is EvalCategory.SAFETY:
        return _safety_checks(case.expected, actual)
    return _trajectory_checks(case.expected, actual)


def _routing_checks(
    expected: RoutingExpectation,
    actual: RoutingActual,
) -> List[EvalCheck]:
    return [
        _check("intent_matches", expected.intent.value, actual.intent.value),
        _check(
            "clarification_matches",
            expected.needs_clarification,
            actual.needs_clarification,
        ),
    ]


def _tool_checks(
    expected: ToolExpectation,
    actual: ToolActual,
) -> List[EvalCheck]:
    names = [call.tool_name for call in actual.calls]
    checks = [
        _check(
            f"required_tool_{name}",
            True,
            name in names,
        )
        for name in expected.required_tool_names
    ]
    checks.extend(
        _check(
            f"forbidden_tool_{name}",
            False,
            name in names,
        )
        for name in expected.forbidden_tool_names
    )
    calls_by_name = {call.tool_name: call for call in actual.calls}
    for name, required_args in expected.required_args.items():
        actual_args = calls_by_name.get(name).tool_args if name in calls_by_name else {}
        checks.append(
            _check(
                f"tool_args_{name}",
                required_args,
                {key: actual_args.get(key) for key in required_args},
            )
        )
    if expected.ordered:
        checks.append(
            _check(
                "tool_order_matches",
                expected.required_tool_names,
                names,
                passed=_contains_in_order(names, expected.required_tool_names),
            )
        )
    return checks


def _rag_checks(expected: RagExpectation, actual: RagActual) -> List[EvalCheck]:
    checks = []
    if expected.status is not None:
        checks.append(
            _check("rag_status_matches", expected.status.value, actual.status.value)
        )
    checks.extend(
        _check(f"required_source_{source}", True, source in actual.sources)
        for source in expected.required_sources
    )
    checks.append(
        _check(
            "minimum_citations",
            f">={expected.min_citations}",
            actual.citation_count,
            passed=actual.citation_count >= expected.min_citations,
        )
    )
    return checks


def _memory_checks(
    expected: MemoryExpectation,
    actual: MemoryActual,
) -> List[EvalCheck]:
    checks = [_check("memory_success", expected.should_succeed, actual.success)]
    checks.extend(
        _check(f"required_memory_{index}", True, fragment in actual.output)
        for index, fragment in enumerate(expected.required_fragments, start=1)
    )
    checks.extend(
        _check(f"forbidden_memory_{index}", False, fragment in actual.output)
        for index, fragment in enumerate(expected.forbidden_fragments, start=1)
    )
    if expected.expected_error_code is not None:
        checks.append(
            _check(
                "memory_error_code",
                expected.expected_error_code,
                actual.error_code,
            )
        )
    return checks


def _safety_checks(
    expected: SafetyExpectation,
    actual: SafetyActual,
) -> List[EvalCheck]:
    checks = [
        _check("safety_outcome", expected.outcome.value, actual.outcome.value)
    ]
    checks.extend(
        _check(f"safety_error_{code}", True, code in actual.error_codes)
        for code in expected.expected_error_codes
    )
    checks.extend(
        _check(f"required_output_{index}", True, fragment in actual.output)
        for index, fragment in enumerate(expected.required_output_fragments, start=1)
    )
    checks.extend(
        _check(f"forbidden_output_{index}", False, fragment in actual.output)
        for index, fragment in enumerate(expected.forbidden_output_fragments, start=1)
    )
    return checks


def _trajectory_checks(
    expected: TrajectoryExpectation,
    actual: TrajectoryActual,
) -> List[EvalCheck]:
    checks = [
        _check(f"required_step_{step}", True, step in actual.steps)
        for step in expected.required_steps
    ]
    checks.extend(
        _check(f"forbidden_step_{step}", False, step in actual.steps)
        for step in expected.forbidden_steps
    )
    if expected.ordered_steps:
        checks.append(
            _check(
                "step_order_matches",
                expected.ordered_steps,
                actual.steps,
                passed=_contains_in_order(actual.steps, expected.ordered_steps),
            )
        )
    return checks


def _check(
    name: str,
    expected: Any,
    actual: Any,
    *,
    passed: Optional[bool] = None,
) -> EvalCheck:
    return EvalCheck(
        name=name,
        passed=expected == actual if passed is None else passed,
        expected=expected,
        actual=actual,
    )


def _contains_in_order(actual: Iterable[str], expected: Iterable[str]) -> bool:
    expected_iterator = iter(expected)
    next_expected = next(expected_iterator, None)
    for item in actual:
        if item == next_expected:
            next_expected = next(expected_iterator, None)
        if next_expected is None:
            return True
    return next_expected is None
