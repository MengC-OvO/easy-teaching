"""Structured output models for the activity Planning Skill."""

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints, model_validator

from app.schemas.graph_state import Citation


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ActivityPlanStep(BaseModel):
    """One ordered, teacher-readable activity instruction."""

    sequence: int = Field(ge=1)
    instruction: NonEmptyText
    duration_minutes: Optional[int] = Field(default=None, ge=1, le=180)


class ActivityObservationPoint(BaseModel):
    """What an educator can observe without diagnosing a child."""

    focus: NonEmptyText
    indicators: List[NonEmptyText] = Field(min_length=1)


class ActivityPlanEylfAlignment(BaseModel):
    """An EYLF alignment grounded in retrieved evidence."""

    outcome: NonEmptyText
    rationale: NonEmptyText
    evidence_ids: List[NonEmptyText] = Field(min_length=1)
    citations: List[Citation] = Field(min_length=1)


class ActivityPlan(BaseModel):
    """Validated draft produced by the Planning Skill."""

    title: NonEmptyText
    class_profile_summary: NonEmptyText
    learning_goals: List[NonEmptyText] = Field(min_length=1)
    materials: List[NonEmptyText] = Field(min_length=1)
    steps: List[ActivityPlanStep] = Field(min_length=1)
    observation_points: List[ActivityObservationPoint] = Field(min_length=1)
    eylf_alignments: List[ActivityPlanEylfAlignment] = Field(min_length=1)
    is_draft: Literal[True] = True

    @model_validator(mode="after")
    def validate_step_sequence(self) -> "ActivityPlan":
        actual = [step.sequence for step in self.steps]
        expected = list(range(1, len(self.steps) + 1))
        if actual != expected:
            raise ValueError(
                "activity plan steps must use contiguous sequence numbers starting at 1"
            )
        return self
