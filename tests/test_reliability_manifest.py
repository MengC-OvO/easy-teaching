import json

import pytest

from evals import load_reliability_scenarios


def test_reliability_manifest_covers_day4_failure_boundaries() -> None:
    scenarios = load_reliability_scenarios()
    ids = {scenario.id for scenario in scenarios}

    assert len(scenarios) == 11
    assert {
        "transient-timeout-retry",
        "rate-limit-and-server-retry",
        "total-time-budget-stop",
        "invalid-json-regeneration",
        "router-timeout-clarification",
        "rag-generation-evidence-fallback",
        "rag-empty-retrieval-clarification",
        "recoverable-tool-error-continue",
        "react-step-budget-stop",
        "safe-error-metadata",
        "sse-trace-allowlist",
    } == ids


def test_reliability_manifest_rejects_duplicate_scenarios(tmp_path) -> None:
    scenarios = [
        scenario.model_dump(mode="json")
        for scenario in load_reliability_scenarios()
    ]
    scenarios.append(scenarios[0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(scenarios), encoding="utf-8")

    with pytest.raises(ValueError, match="ids must be unique"):
        load_reliability_scenarios(path)
