"""Load and validate offline evaluation fixtures without running the Agent."""

import json
from pathlib import Path
from typing import List

from evals.schemas import EvalCase, EvalCategory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "evals" / "agent_cases.json"


def load_eval_cases(path: Path = DEFAULT_CASES_PATH) -> List[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation fixture must be a JSON list")

    cases = [EvalCase.model_validate(item) for item in payload]
    ids = [case.id for case in cases]
    duplicate_ids = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate evaluation case ids: {', '.join(duplicate_ids)}")

    missing_categories = set(EvalCategory) - {case.category for case in cases}
    if missing_categories:
        names = ", ".join(sorted(category.value for category in missing_categories))
        raise ValueError(f"Evaluation fixture is missing categories: {names}")
    return cases
