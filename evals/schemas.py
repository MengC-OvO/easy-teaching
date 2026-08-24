"""Validated contracts for the offline Week 4 evaluation harness."""

from enum import Enum
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.schemas import Intent


EvalIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9_.-]*$",
    ),
]
EvalText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
]


class EvalContract(BaseModel):
    """Reject misspelled fields in fixtures and generated evaluation results."""

    model_config = ConfigDict(extra="forbid")


class EvalCategory(str, Enum):
    ROUTING = "routing"
    TOOL = "tool"
    RAG = "rag"
    MEMORY = "memory"
    SAFETY = "safety"
    TRAJECTORY = "trajectory"


class EvalMode(str, Enum):
    OFFLINE = "offline"
    LIVE_MODEL = "live_model"


class MemoryEvalTarget(str, Enum):
    PROFILE_CONTEXT = "profile_context"
    RECALL_TOOL = "recall_tool"


class SafetyOutcome(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


class RagStatus(str, Enum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"


class EvalInput(EvalContract):
    """Synthetic request scope supplied to the system under evaluation."""

    message: EvalText
    teacher_id: Optional[EvalIdentifier] = None
    class_id: Optional[EvalIdentifier] = None
    conversation_context: str = Field(default="", max_length=20_000)


class RoutingExpectation(EvalContract):
    intent: Intent
    needs_clarification: bool = False


class ToolExpectation(EvalContract):
    required_tool_names: List[EvalIdentifier] = Field(min_length=1)
    forbidden_tool_names: List[EvalIdentifier] = Field(default_factory=list)
    required_args: Dict[EvalIdentifier, Dict[str, Any]] = Field(default_factory=dict)
    ordered: bool = False

    @model_validator(mode="after")
    def validate_tool_sets(self) -> "ToolExpectation":
        _require_disjoint(
            self.required_tool_names,
            self.forbidden_tool_names,
            label="tool names",
        )
        unknown_arg_tools = set(self.required_args) - set(self.required_tool_names)
        if unknown_arg_tools:
            names = ", ".join(sorted(unknown_arg_tools))
            raise ValueError(f"required_args refers to non-required tools: {names}")
        return self


class RagExpectation(EvalContract):
    status: Optional[RagStatus] = None
    required_sources: List[EvalIdentifier] = Field(default_factory=list)
    min_citations: int = Field(default=0, ge=0)


class MemoryExpectation(EvalContract):
    target: MemoryEvalTarget
    should_succeed: bool = True
    required_fragments: List[EvalText] = Field(default_factory=list)
    forbidden_fragments: List[EvalText] = Field(default_factory=list)
    expected_error_code: Optional[EvalIdentifier] = None

    @model_validator(mode="after")
    def validate_memory_fragments(self) -> "MemoryExpectation":
        _require_disjoint(
            self.required_fragments,
            self.forbidden_fragments,
            label="memory fragments",
        )
        return self


class SafetyExpectation(EvalContract):
    outcome: SafetyOutcome
    expected_error_codes: List[EvalIdentifier] = Field(default_factory=list)
    required_output_fragments: List[EvalText] = Field(default_factory=list)
    forbidden_output_fragments: List[EvalText] = Field(default_factory=list)


class TrajectoryExpectation(EvalContract):
    required_steps: List[EvalIdentifier] = Field(default_factory=list)
    forbidden_steps: List[EvalIdentifier] = Field(default_factory=list)
    ordered_steps: List[EvalIdentifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trajectory_steps(self) -> "TrajectoryExpectation":
        _require_disjoint(
            self.required_steps,
            self.forbidden_steps,
            label="trajectory steps",
        )
        missing_required = set(self.ordered_steps) - set(self.required_steps)
        if missing_required:
            names = ", ".join(sorted(missing_required))
            raise ValueError(f"ordered_steps must also be required_steps: {names}")
        return self


EvalExpectation = Union[
    RoutingExpectation,
    ToolExpectation,
    RagExpectation,
    MemoryExpectation,
    SafetyExpectation,
    TrajectoryExpectation,
]


_EXPECTATION_TYPES = {
    EvalCategory.ROUTING: RoutingExpectation,
    EvalCategory.TOOL: ToolExpectation,
    EvalCategory.RAG: RagExpectation,
    EvalCategory.MEMORY: MemoryExpectation,
    EvalCategory.SAFETY: SafetyExpectation,
    EvalCategory.TRAJECTORY: TrajectoryExpectation,
}


class EvalCase(EvalContract):
    """One deterministic fixture with a typed expected outcome."""

    id: EvalIdentifier
    category: EvalCategory
    input: EvalInput
    expected: EvalExpectation
    tags: List[EvalIdentifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def expectation_matches_category(self) -> "EvalCase":
        expected_type = _EXPECTATION_TYPES[self.category]
        if not isinstance(self.expected, expected_type):
            raise ValueError(
                f"{self.category.value} cases require "
                f"{expected_type.__name__}"
            )
        return self


class EvalCheck(EvalContract):
    """One human-readable assertion produced by a future evaluator."""

    name: EvalIdentifier
    passed: bool
    expected: Any = None
    actual: Any = None
    message: Optional[str] = Field(default=None, max_length=2_000)


class RoutingActual(EvalContract):
    intent: Intent
    needs_clarification: bool


class ObservedToolCall(EvalContract):
    tool_name: EvalIdentifier
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class ToolActual(EvalContract):
    calls: List[ObservedToolCall] = Field(default_factory=list)


class RagActual(EvalContract):
    status: RagStatus
    sources: List[EvalIdentifier] = Field(default_factory=list)
    citation_count: int = Field(default=0, ge=0)


class MemoryActual(EvalContract):
    success: bool
    output: str = ""
    error_code: Optional[EvalIdentifier] = None


class SafetyActual(EvalContract):
    outcome: SafetyOutcome
    output: str = ""
    error_codes: List[EvalIdentifier] = Field(default_factory=list)


class TrajectoryActual(EvalContract):
    steps: List[EvalIdentifier] = Field(default_factory=list)


EvalActual = Union[
    RoutingActual,
    ToolActual,
    RagActual,
    MemoryActual,
    SafetyActual,
    TrajectoryActual,
]


class EvalTokenUsage(EvalContract):
    model_calls: int = Field(default=0, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def total_covers_known_tokens(self) -> "EvalTokenUsage":
        known_total = self.prompt_tokens + self.completion_tokens
        if self.total_tokens < known_total:
            raise ValueError(
                "total_tokens cannot be less than prompt_tokens + completion_tokens"
            )
        return self


class EvalResult(EvalContract):
    """Normalized result returned by every category-specific evaluator."""

    case_id: EvalIdentifier
    category: EvalCategory
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    checks: List[EvalCheck] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    token_usage: EvalTokenUsage = Field(default_factory=EvalTokenUsage)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    error_code: Optional[EvalIdentifier] = None


class EvalCategorySummary(EvalContract):
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    average_score: float = Field(ge=0.0, le=1.0)


class EvalRunSummary(EvalContract):
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    average_score: float = Field(ge=0.0, le=1.0)
    p50_latency_ms: float = Field(ge=0.0)
    p95_latency_ms: float = Field(ge=0.0)
    token_usage: EvalTokenUsage = Field(default_factory=EvalTokenUsage)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    categories: Dict[EvalCategory, EvalCategorySummary] = Field(default_factory=dict)


class EvalReport(EvalContract):
    mode: EvalMode
    generated_at: datetime
    summary: EvalRunSummary
    results: List[EvalResult]


def _require_disjoint(required: List[str], forbidden: List[str], *, label: str) -> None:
    overlap = set(required) & set(forbidden)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"Required and forbidden {label} overlap: {names}")
