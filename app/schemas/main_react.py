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


class TaskType(str, Enum):
    """Semantic request contract selected by Main, not by keyword routing."""

    GENERAL = "general"
    ACTIVITY_PLAN = "activity_plan"
    SAFETY_REVIEW = "safety_review"
    CLASS_CONTEXT = "class_context"
    POLICY_QA = "policy_qa"
    RECORD_QUERY = "record_query"
    OBSERVATION_DRAFT = "observation_draft"
    EDUCATIONAL_RECORD_DRAFT = "educational_record_draft"
    FAMILY_COMMUNICATION = "family_communication"
    CONTROLLED_WRITE = "controlled_write"


class CompletionAction(str, Enum):
    """Explicit user-requested side effects that must precede a final answer.

    This is a completion contract, not a step-by-step plan: Main remains free to
    choose any ReAct trajectory, while validation prevents it from silently
    stopping after only the read or drafting portion of a requested operation.
    """

    SAVE_OBSERVATION = "save_observation"
    SAVE_EDUCATIONAL_RECORD = "save_educational_record"
    EXPORT_RECORDS = "export_records"
    UPLOAD_EXPORT_TO_GOOGLE_DRIVE = "upload_export_to_google_drive"


class CapabilityCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result_key: str = Field(min_length=1)

    def signature(self) -> str:
        """用于检测忽略大小写和多余空白后的重复调用。"""

        return json.dumps(
            {
                "name": self.name,
                "arguments": _normalize_signature_value(self.arguments),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


class WorkerName(str, Enum):
    CURRICULUM_RESEARCH = "curriculum_research_worker"
    RECORD_CONTEXT = "record_context_worker"


class WorkerCall(BaseModel):
    name: WorkerName
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result_key: str = Field(min_length=1)

    def signature(self) -> str:
        return json.dumps(
            {"name": self.name.value, "arguments": self.arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )


class MainDecision(BaseModel):
    task_type: TaskType = TaskType.GENERAL
    artifact_title: Optional[str] = Field(default=None, min_length=1, max_length=160)
    requires_activity_safety: bool = Field(
        default=False,
        description=(
            "True only for a proposed future activity/learning experience or an "
            "explicit activity-risk review; false for policy Q&A, observations, "
            "records, family messages, exports, and retrospective events."
        ),
    )
    completion_actions: List[CompletionAction] = Field(
        default_factory=list,
        description=(
            "Deprecated checkpoint/schema compatibility field. Runtime completion "
            "requirements are derived independently from registered controlled "
            "Tools and never trust this model-supplied value."
        ),
    )
    reason: str = Field(min_length=1)
    tool_calls: List[CapabilityCall] = Field(default_factory=list)
    worker_calls: List[WorkerCall] = Field(default_factory=list)
    final_answer: Optional[str] = None
    clarification_question: Optional[str] = None

    @model_validator(mode="after")
    def validate_single_decision_kind(self) -> "MainDecision":
        # Keep a malformed safety flag from turning an unrelated request into an
        # eight-turn feedback loop. This is consistency validation on one model
        # decision, not a workflow plan or keyword router.
        if self.task_type not in {TaskType.ACTIVITY_PLAN, TaskType.SAFETY_REVIEW}:
            self.requires_activity_safety = False
        if not self.final_answer:
            self.artifact_title = None
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


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, dict):
        return {
            str(key): _normalize_signature_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_normalize_signature_value(item) for item in value]
    return value

