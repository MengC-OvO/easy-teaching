"""Shared contracts between the main graph and specialist workflows."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.graph_state import (
    Approval,
    ApprovalStatus,
    Citation,
    Draft,
    GraphError,
    GraphState,
    SafetyFlag,
    TraceEvent,
    WorkflowStatus,
)


class SpecialistKind(str, Enum):
    PLANNING = "planning"
    DOCUMENTATION = "documentation"
    POLICY = "policy"
    FAMILY = "family"


class SpecialistInput(BaseModel):
    """The bounded state every specialist workflow may receive."""

    specialist: SpecialistKind
    request_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    thread_id: Optional[str] = None
    user_message: str = Field(min_length=1)
    teacher_id: Optional[str] = None
    class_id: Optional[str] = None
    conversation_context: str = ""
    approval: Approval = Field(default_factory=Approval)

    @classmethod
    def from_graph_state(
        cls,
        state: GraphState,
        *,
        specialist: SpecialistKind,
        conversation_context: str = "",
    ) -> "SpecialistInput":
        return cls(
            specialist=specialist,
            request_id=state.request_id,
            session_id=state.session_id,
            thread_id=state.thread_id,
            user_message=state.user_message,
            teacher_id=state.teacher_id,
            class_id=state.class_id,
            conversation_context=conversation_context,
            approval=state.approval,
        )


class SpecialistResult(BaseModel):
    """The result shape every specialist workflow returns to the main graph."""

    specialist: SpecialistKind
    status: WorkflowStatus
    draft: Optional[Draft] = None
    citations: List[Citation] = Field(default_factory=list)
    approval: Approval = Field(default_factory=Approval)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    errors: List[GraphError] = Field(default_factory=list)
    safety_flags: List[SafetyFlag] = Field(default_factory=list)
    trace: List[TraceEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_payload(self) -> "SpecialistResult":
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("clarification result must include a question")
        if self.clarification_question and not self.needs_clarification:
            raise ValueError("clarification question requires needs_clarification=true")
        if (
            self.status is WorkflowStatus.WAITING_FOR_APPROVAL
            and self.approval.status is not ApprovalStatus.REQUIRED
        ):
            raise ValueError("waiting-for-approval result must require approval")
        if self.status is WorkflowStatus.FAILED and not self.errors:
            raise ValueError("failed specialist result must include an error")
        return self

    def to_graph_update(self) -> Dict[str, Any]:
        """Return only the fields a specialist is allowed to update on GraphState."""
        return {
            "workflow_status": self.status,
            "draft": self.draft,
            "citations": self.citations,
            "approval": self.approval,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "errors": self.errors,
            "safety_flags": self.safety_flags,
            "trace": self.trace,
        }
