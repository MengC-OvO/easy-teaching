"""Contracts for privacy-bounded, cross-thread long-term memory.

Only stable teacher preferences, confirmed class facts, and durable cross-session
constraints may enter the long-term memory pipeline. Conversation transcripts,
draft text, policy answers, tool payloads, and child/family personal data are
intentionally outside this schema.
"""

from enum import Enum

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class LongTermMemoryScope(str, Enum):
    TEACHER = "teacher"
    CLASS = "class"


class LongTermMemoryType(str, Enum):
    TEACHER_PREFERENCE = "teacher_preference"
    CLASS_FACT = "class_fact"
    LONG_TERM_CONSTRAINT = "long_term_constraint"


class LongTermMemoryAction(str, Enum):
    """The only operations an LLM may request against durable memory."""

    NOOP = "noop"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


class MemoryRetrievalMode(str, Enum):
    """Controls whether memory is automatic profile context or on-demand only."""

    PROFILE = "profile"
    RECALL_ONLY = "recall_only"


class LongTermMemoryCandidate(BaseModel):
    """A proposed durable fact, ready for later review and persistence."""

    scope: LongTermMemoryScope
    scope_id: str = Field(min_length=1, max_length=120)
    memory_type: LongTermMemoryType
    content: str = Field(min_length=1, max_length=400)
    reason: str = Field(min_length=1, max_length=240)
    retrieval_mode: MemoryRetrievalMode = MemoryRetrievalMode.RECALL_ONLY
    importance: int = Field(default=2, ge=1, le=5)

    @model_validator(mode="after")
    def validate_scope(self) -> "LongTermMemoryCandidate":
        if (
            self.memory_type is LongTermMemoryType.TEACHER_PREFERENCE
            and self.scope is not LongTermMemoryScope.TEACHER
        ):
            raise ValueError("teacher_preference memory must use teacher scope")
        if (
            self.memory_type is LongTermMemoryType.CLASS_FACT
            and self.scope is not LongTermMemoryScope.CLASS
        ):
            raise ValueError("class_fact memory must use class scope")
        if (
            self.retrieval_mode is MemoryRetrievalMode.PROFILE
            and (
                self.scope is not LongTermMemoryScope.TEACHER
                or self.memory_type is not LongTermMemoryType.TEACHER_PREFERENCE
            )
        ):
            raise ValueError(
                "profile memory must be a teacher-scoped teacher_preference"
            )
        return self


class LongTermMemoryOperation(BaseModel):
    """A proposed memory mutation, applied only after store-side validation."""

    action: LongTermMemoryAction
    memory_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    candidate: Optional[LongTermMemoryCandidate] = None
    reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_operation(self) -> "LongTermMemoryOperation":
        if self.action is LongTermMemoryAction.NOOP:
            if self.memory_id is not None or self.candidate is not None:
                raise ValueError("noop memory operation cannot target a memory")
        elif self.action is LongTermMemoryAction.INSERT:
            if self.candidate is None or self.memory_id is not None:
                raise ValueError("insert memory operation requires a candidate only")
        elif self.action is LongTermMemoryAction.UPDATE:
            if self.memory_id is None or self.candidate is None:
                raise ValueError("update memory operation requires memory_id and candidate")
        elif self.action is LongTermMemoryAction.DELETE:
            if self.memory_id is None or self.candidate is not None:
                raise ValueError("delete memory operation requires memory_id only")
        return self


class LongTermMemoryDecision(BaseModel):
    """A bounded, structured decision for one completed graph turn."""

    operations: List[LongTermMemoryOperation] = Field(default_factory=list, max_length=4)
