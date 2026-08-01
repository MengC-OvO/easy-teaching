"""Validated manifest for the deterministic Day 4 fault-injection suite."""

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Annotated


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELIABILITY_PATH = (
    PROJECT_ROOT / "data" / "evals" / "reliability_scenarios.json"
)
ScenarioId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    ),
]


class ReliabilityScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: ScenarioId
    fault: str = Field(min_length=1, max_length=500)
    expected: str = Field(min_length=1, max_length=500)
    test_node: str = Field(
        min_length=1,
        max_length=500,
        pattern=r"^tests/[a-z0-9_/]+\.py::test_[a-z0-9_]+$",
    )

    @model_validator(mode="after")
    def require_fault_test_name(self) -> "ReliabilityScenario":
        if not any(
            marker in self.test_node
            for marker in (
                "retry",
                "retries",
                "fallback",
                "falls_back",
                "clarif",
                "error",
                "budget",
                "safe",
            )
        ):
            raise ValueError("reliability scenario must reference a fault-behaviour test")
        return self


def load_reliability_scenarios(
    path: Path = DEFAULT_RELIABILITY_PATH,
) -> List[ReliabilityScenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Reliability scenario manifest must be a JSON list")
    scenarios = [ReliabilityScenario.model_validate(item) for item in payload]
    ids = [scenario.id for scenario in scenarios]
    nodes = [scenario.test_node for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Reliability scenario ids must be unique")
    if len(nodes) != len(set(nodes)):
        raise ValueError("Reliability scenario test nodes must be unique")
    if len(scenarios) < 10:
        raise ValueError("Reliability suite must cover at least ten fault scenarios")
    return scenarios
