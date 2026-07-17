import operator
from enum import Enum
from typing import Any, Dict, List, Optional
from typing_extensions import Annotated

from pydantic import BaseModel, Field


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
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"
    APPROVED = "approved"
    REJECTED = "rejected"


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


class GraphState(BaseModel):
    request_id: str
    session_id: str
    user_message: str
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
