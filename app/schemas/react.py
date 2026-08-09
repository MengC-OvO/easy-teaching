from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator

class ReActAction(str, Enum):
    CALL_TOOL = "call_tool"
    FINAL_ANSWER = "final_answer"


class ToolCall(BaseModel):
    tool_name: str = Field(min_length=1)
    tool_args: Dict[str, Any] = Field(default_factory=dict)


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
