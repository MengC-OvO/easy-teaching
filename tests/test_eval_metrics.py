from evals import (
    EvalCategory,
    EvalCheck,
    EvalMode,
    EvalResult,
    EvalTokenUsage,
    build_eval_report,
)


def _result(case_id: str, category: EvalCategory, *, passed: bool, latency: float):
    return EvalResult(
        case_id=case_id,
        category=category,
        passed=passed,
        score=1.0 if passed else 0.5,
        checks=[EvalCheck(name="check", passed=passed)],
        latency_ms=latency,
        token_usage=EvalTokenUsage(
            model_calls=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        estimated_cost_usd=0.01,
    )


def test_report_aggregates_results_and_categories() -> None:
    report = build_eval_report(
        [
            _result("routing-pass", EvalCategory.ROUTING, passed=True, latency=10),
            _result("rag-fail", EvalCategory.RAG, passed=False, latency=30),
        ],
        mode=EvalMode.LIVE_MODEL,
    )

    assert report.mode is EvalMode.LIVE_MODEL
    assert report.summary.total == 2
    assert report.summary.pass_rate == 0.5
    assert report.summary.average_score == 0.75
    assert report.summary.token_usage.total_tokens == 30
    assert report.summary.estimated_cost_usd == 0.02
    assert report.summary.categories[EvalCategory.ROUTING].pass_rate == 1.0
    assert report.summary.categories[EvalCategory.RAG].pass_rate == 0.0


def test_empty_report_has_zero_metrics() -> None:
    report = build_eval_report([])

    assert report.summary.total == 0
    assert report.summary.pass_rate == 0.0
    assert report.summary.p95_latency_ms == 0.0
