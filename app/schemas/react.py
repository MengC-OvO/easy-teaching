import operator
from enum import Enum
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, Field, model_validator


class ReActAction(str, Enum):
    CALL_TOOL = "call_tool"
    FINAL_ANSWER = "final_answer"


class StopReason(str, Enum):
    NOT_STOPPED = "not_stopped"
    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    TOOL_ERROR = "tool_error"
    APPROVAL_REQUIRED = "approval_required"
    MODEL_ERROR = "model_error"


class ToolCall(BaseModel):
    tool_name: str = Field(min_length=1)
    tool_args: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    tool_name: str = Field(min_length=1)
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None


class ReActDecision(BaseModel):
    action: ReActAction
    reason: str = Field(min_length=1)
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> "ReActDecision":
        if self.action is ReActAction.CALL_TOOL:
            if self.tool_call is None:
                raise ValueError("tool_call is required when action is call_tool")
            if self.final_answer is not None:
                raise ValueError("final_answer must be empty when action is call_tool")
        if self.action is ReActAction.FINAL_ANSWER:
            if not self.final_answer:
                raise ValueError("final_answer is required when action is final_answer")
            if self.tool_call is not None:
                raise ValueError("tool_call must be empty when action is final_answer")
        return self


class ReActState(BaseModel):
    user_message: str
    teacher_id: Optional[str] = None
    class_id: Optional[str] = None
    conversation_context: str = ""
    max_steps: int = Field(default=4, ge=1)
    current_step: int = Field(default=0, ge=0)
    decision: Optional[ReActDecision] = None
    final_answer: Optional[str] = None
    stop_reason: StopReason = StopReason.NOT_STOPPED
    observations: Annotated[List[Observation], operator.add] = Field(default_factory=list)

    @property
    def should_stop(self) -> bool:
        return self.stop_reason is not StopReason.NOT_STOPPED

    @property
    def has_steps_remaining(self) -> bool:
        return self.current_step < self.max_steps
