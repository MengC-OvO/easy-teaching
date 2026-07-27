"""API contracts for the learning-record drafting flow.

The intake contract is deliberately transient: accepting an observation must
not create a database record or add the observation text to an application
trace. A later approval step will be the only path that persists a record.
"""

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, StringConstraints, field_validator
from typing_extensions import Annotated


ObservationText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]


class PIIType(str, Enum):
    """High-confidence sensitive-data categories handled locally."""

    EMAIL = "email"
    PHONE = "phone"
    PERSON_NAME = "person_name"


class DeidentifiedObservation(BaseModel):
    """The only observation representation safe to pass to a model provider."""

    safe_text: ObservationText
    redacted_types: list[PIIType] = Field(default_factory=list)
    replacement_count: int = Field(default=0, ge=0)

    @field_validator("redacted_types")
    @classmethod
    def redacted_types_must_be_unique(
        cls,
        values: list[PIIType],
    ) -> list[PIIType]:
        if len(values) != len(set(values)):
            raise ValueError("redacted_types must not contain duplicates")
        return values


class LearningRecordDraft(BaseModel):
    """A teacher-reviewable draft generated from a de-identified observation."""

    observation_summary: ObservationText
    learning_analysis: ObservationText
    possible_next_steps: list[ObservationText] = Field(min_length=1, max_length=5)
    is_draft: Literal[True] = True
