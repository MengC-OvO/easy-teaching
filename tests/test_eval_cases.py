import json
from collections import Counter

import pytest

from evals import EvalCategory, load_eval_cases


def test_week4_fixture_loads_five_cases_for_every_category() -> None:
    cases = load_eval_cases()

    assert len(cases) == 30
    assert Counter(case.category for case in cases) == {
        category: 5 for category in EvalCategory
    }
    assert len({case.id for case in cases}) == len(cases)


def test_week4_fixture_round_trips_through_the_eval_contracts() -> None:
    cases = load_eval_cases()

    payload = [case.model_dump(mode="json") for case in cases]

    assert payload[0]["id"] == "routing-activity-planning"
    assert payload[-1]["id"] == "trajectory-clarification"
    assert all(item["input"]["message"] for item in payload)


def test_eval_loader_rejects_duplicate_case_ids(tmp_path) -> None:
    cases = [case.model_dump(mode="json") for case in load_eval_cases()]
    cases[1]["id"] = cases[0]["id"]
    path = tmp_path / "duplicates.json"
    path.write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate evaluation case ids"):
        load_eval_cases(path)


def test_eval_loader_requires_every_category(tmp_path) -> None:
    cases = [
        case.model_dump(mode="json")
        for case in load_eval_cases()
        if case.category is not EvalCategory.SAFETY
    ]
    path = tmp_path / "missing-category.json"
    path.write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(ValueError, match="missing categories: safety"):
        load_eval_cases(path)
