"""Main ReAct 每轮决策和执行结果的结构化契约。"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CapabilitySource(str, Enum):
    TOOL = "tool"
    MCP = "mcp"
    WORKER = "worker"
    SYSTEM = "system"


class ObservationStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT = "insufficient"
    FAILED = "failed"
    REJECTED = "rejected"


class CapabilityCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    needs: List[str] = Field(default_factory=list)
    result_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def normalize_dependencies(self) -> "CapabilityCall":
        self.needs = list(dict.fromkeys(self.needs))
        return self

    def signature(self) -> str:
        """用于检测模型是否重复请求完全相同的调用。"""

        return json.dumps(
            {"name": self.name, "arguments": self.arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


class WorkerName(str, Enum):
    INTERNAL_RESEARCH = "internal_research_worker"
    LOCAL_CONTEXT = "local_context_worker"
    EXTERNAL_RESEARCH = "external_research_worker"


class WorkerCall(BaseModel):
    name: WorkerName
    arguments: Dict[str, Any] = Field(default_factory=dict)
    needs: List[str] = Field(default_factory=list)
    result_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def normalize_dependencies(self) -> "WorkerCall":
        self.needs = list(dict.fromkeys(self.needs))
        return self

    def signature(self) -> str:
        return json.dumps(
            {"name": self.name.value, "arguments": self.arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


class MainDecision(BaseModel):
    reason: str = Field(min_length=1)
    tool_calls: List[CapabilityCall] = Field(default_factory=list)
    worker_calls: List[WorkerCall] = Field(default_factory=list)
    final_answer: Optional[str] = None
    clarification_question: Optional[str] = None

    @model_validator(mode="after")
    def validate_single_decision_kind(self) -> "MainDecision":
        choices = [
            bool(self.tool_calls),
            bool(self.worker_calls),
            bool(self.final_answer),
            bool(self.clarification_question),
        ]
        if sum(choices) != 1:
            raise ValueError(
                "MainDecision must choose exactly one of tool_calls, "
                "worker_calls, final_answer, or clarification_question"
            )

        result_keys = [call.result_key for call in self.current_calls]
        if len(result_keys) != len(set(result_keys)):
            raise ValueError("result_key values must be unique in one decision")
        return self

    @property
    def current_calls(self) -> List[Any]:
        if self.tool_calls:
            return list(self.tool_calls)
        return list(self.worker_calls)


class CapabilityObservation(BaseModel):
    result_key: str = Field(min_length=1)
    capability_name: str = Field(min_length=1)
    source_kind: CapabilitySource
    status: ObservationStatus
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None

    @property
    def is_available(self) -> bool:
        return self.status in {
            ObservationStatus.COMPLETED,
            ObservationStatus.INSUFFICIENT,
        }

