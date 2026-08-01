"""Aggregate deterministic and live evaluation results into one report."""

from datetime import datetime, timezone
from typing import Iterable, List

from evals.schemas import (
    EvalCategory,
    EvalCategorySummary,
    EvalMode,
    EvalReport,
    EvalResult,
    EvalRunSummary,
    EvalTokenUsage,
)


def build_eval_report(
    results: Iterable[EvalResult],
    *,
    mode: EvalMode = EvalMode.OFFLINE,
) -> EvalReport:
    resolved = list(results)
    return EvalReport(
        mode=mode,
        generated_at=datetime.now(timezone.utc),
        summary=_summary(resolved),
        results=resolved,
    )


def _summary(results: List[EvalResult]) -> EvalRunSummary:
    total = len(results)
    passed = sum(result.passed for result in results)
    latencies = sorted(result.latency_ms for result in results)
    usage = EvalTokenUsage(
        model_calls=sum(result.token_usage.model_calls for result in results),
        prompt_tokens=sum(result.token_usage.prompt_tokens for result in results),
        completion_tokens=sum(
            result.token_usage.completion_tokens for result in results
        ),
        total_tokens=sum(result.token_usage.total_tokens for result in results),
    )
    categories = {
        category: _category_summary(
            [result for result in results if result.category is category]
        )
        for category in EvalCategory
        if any(result.category is category for result in results)
    }
    return EvalRunSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 0.0,
        average_score=(
            sum(result.score for result in results) / total if total else 0.0
        ),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        token_usage=usage,
        estimated_cost_usd=sum(
            result.estimated_cost_usd for result in results
        ),
        categories=categories,
    )


def _category_summary(results: List[EvalResult]) -> EvalCategorySummary:
    total = len(results)
    passed = sum(result.passed for result in results)
    return EvalCategorySummary(
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=passed / total if total else 0.0,
        average_score=(
            sum(result.score for result in results) / total if total else 0.0
        ),
    )


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * percentile))))
    return values[index]
