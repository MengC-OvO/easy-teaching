import json

import pytest

from evals import load_reliability_scenarios


def test_reliability_manifest_covers_day4_failure_boundaries() -> None:
    scenarios = load_reliability_scenarios()
    ids = {scenario.id for scenario in scenarios}

    assert len(scenarios) == 12
    assert {
        "async-provider-retry",
        "nonrecoverable-no-retry",
        "async-tool-timeout",
        "worker-permission-error",
        "main-react-model-fallback",
        "rag-generation-evidence-fallback",
        "rag-empty-retrieval-clarification",
        "invalid-dependency-feedback",
        "main-react-step-budget-stop",
        "safe-error-metadata",
        "async-session-busy-error",
        "parallel-worker-partial-error",
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
