import operator
from enum import Enum
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, Field, model_validator


class Intent(str, Enum):
    UNKNOWN = "unknown"
    ACTIVITY_PLANNING = "activity_planning"
    LEARNING_RECORD = "learning_record"
    POLICY_QA = "policy_qa"
    FAMILY_COMMUNICATION = "family_communication"


class WorkflowStatus(str, Enum):
    CREATED = "created"
    ROUTED = "routed"
    DRAFTING = "drafting"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVED_PENDING_SAVE = "approved_pending_save"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class RiskLevel(str, Enum):
    L0_READ_ONLY = "L0_read_only"
    L1_DRAFT = "L1_draft"
    L2_CONTROLLED_WRITE = "L2_controlled_write"
    L3_FORBIDDEN = "L3_forbidden"


class Citation(BaseModel):
    source: str
    title: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None


class Draft(BaseModel):
    title: Optional[str] = None
    content: str = ""
    is_draft: bool = True


class Approval(BaseModel):
    status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    risk_level: RiskLevel = RiskLevel.L1_DRAFT
    reason: Optional[str] = None
    action_id: Optional[str] = None
    tool_name: Optional[str] = None
    preview: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    step: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphError(BaseModel):
    code: str
    message: str
    recoverable: bool = True


class SafetyFlag(BaseModel):
    code: str
    message: str
    risk_level: RiskLevel


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationTurn(BaseModel):
    role: ConversationRole
    content: str = Field(min_length=1)
    intent: Optional[Intent] = None
    workflow_status: Optional[WorkflowStatus] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextBudget(BaseModel):
    max_recent_turns: int = Field(default=4, ge=0)
    max_recent_tokens: int = Field(default=1200, ge=0)
    max_trace_events: int = Field(default=8, ge=0)
    max_memory_summary_chars: int = Field(default=1500, ge=200)
    max_memory_items: int = Field(default=8, ge=0)


class ConversationMemory(BaseModel):
    """LLM-maintained semantic memory for one conversation thread.

    This deliberately excludes operational state such as the current draft,
    approval, citations, and trace. Those remain canonical on GraphState.
    """

    conversation_goal: Optional[str] = None
    important_requirements: List[str] = Field(default_factory=list)
    confirmed_preferences: List[str] = Field(default_factory=list)
    completed_work: List[str] = Field(default_factory=list)
    open_tasks: List[str] = Field(default_factory=list)
    compact_summary: str = ""


class ThreadContext(BaseModel):
    thread_id: Optional[str] = None
    recent_turns: Annotated[List[ConversationTurn], operator.add] = Field(
        default_factory=list
    )
    memory: ConversationMemory = Field(default_factory=ConversationMemory)
    tool_trace_summary: Annotated[List[TraceEvent], operator.add] = Field(
        default_factory=list
    )
    budget: ContextBudget = Field(default_factory=ContextBudget)

    @model_validator(mode="after")
    def apply_budget(self) -> "ThreadContext":
        if self.budget.max_recent_turns:
            self.recent_turns = self.recent_turns[-self.budget.max_recent_turns :]
        else:
            self.recent_turns = []

        if self.budget.max_trace_events:
            self.tool_trace_summary = self.tool_trace_summary[
                -self.budget.max_trace_events :
            ]
        else:
            self.tool_trace_summary = []

        self.memory.compact_summary = self.memory.compact_summary[
            : self.budget.max_memory_summary_chars
        ]
        for field_name in (
            "important_requirements",
            "confirmed_preferences",
            "completed_work",
            "open_tasks",
        ):
            values = getattr(self.memory, field_name)
            deduped_values = list(dict.fromkeys(values))
            if self.budget.max_memory_items:
                deduped_values = deduped_values[-self.budget.max_memory_items :]
            else:
                deduped_values = []
            setattr(self.memory, field_name, deduped_values)

        return self


class GraphState(BaseModel):
    request_id: str
    session_id: str
    user_message: str
    thread_id: Optional[str] = None
    teacher_id: Optional[str] = None
    class_id: Optional[str] = None
    # Opaque gateway handle only; plaintext mappings never enter checkpoints.
    privacy_mapping_id: Optional[str] = None
    context: ThreadContext = Field(default_factory=ThreadContext)
    intent: Intent = Intent.UNKNOWN
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    workflow_status: WorkflowStatus = WorkflowStatus.CREATED
    draft: Optional[Draft] = None
    citations: Annotated[List[Citation], operator.add] = Field(default_factory=list)
    approval: Approval = Field(default_factory=Approval)
    trace: Annotated[List[TraceEvent], operator.add] = Field(default_factory=list)
    errors: Annotated[List[GraphError], operator.add] = Field(default_factory=list)
    safety_flags: Annotated[List[SafetyFlag], operator.add] = Field(default_factory=list)
    # Main ReAct 新路径字段。旧字段暂时保留，用于读取历史 checkpoint。
    decision: Optional["MainDecision"] = None
    execution_route: Optional[str] = None
    validation_feedback: Optional["CapabilityObservation"] = None
    observations: Dict[str, "CapabilityObservation"] = Field(default_factory=dict)
    pending_observations: Annotated[
        List["CapabilityObservation"], operator.add
    ] = Field(default_factory=list)
    merged_observation_count: int = Field(default=0, ge=0)
    react_step: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    worker_batch_count: int = Field(default=0, ge=0)
    repeated_call_counts: Dict[str, int] = Field(default_factory=dict)
    tool_attempt_counts: Dict[str, int] = Field(default_factory=dict)
    required_completion_actions: List[str] = Field(default_factory=list)
    # Deprecated checkpoint compatibility only. Runtime draft identity comes from
    # the full loaded_draft_references mapping derived from read observations.
    selected_draft_request_id: Optional[str] = None
    available_tool_names: List[str] = Field(default_factory=list)
    run_trace_start: int = Field(default=0, ge=0)
    run_citation_start: int = Field(default=0, ge=0)


from app.schemas.main_react import (  # noqa: E402 解决双向类型引用
    CapabilityObservation,
    MainDecision,
)

GraphState.model_rebuild()
