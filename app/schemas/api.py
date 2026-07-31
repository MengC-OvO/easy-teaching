"""Validated HTTP and SSE contracts for the EduFlow public API.

These models deliberately contain no FastAPI or LangGraph execution logic.
They define the boundary between an API caller and the existing workflow
domain models so later endpoint work can depend on one stable contract.
"""

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.approval_resume import ApprovalDecision, ApprovalResumeCommand
from app.schemas.graph_state import Approval, Citation, Draft


ApiIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
ApiMessage = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000),
]


class ApiContractModel(BaseModel):
    """Base policy for public API payloads.

    Unknown fields are rejected so a misspelled client field cannot be silently
    ignored at the HTTP boundary.
    """

    model_config = ConfigDict(extra="forbid")


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"


class RunStatus(str, Enum):
    """API-facing lifecycle for one accepted teacher message."""

    ACCEPTED = "accepted"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StreamEventType(str, Enum):
    """Stable SSE event names exposed to API consumers."""

    RUN_STARTED = "run_started"
    ROUTE_SELECTED = "route_selected"
    TRACE = "trace"
    DRAFT_READY = "draft_ready"
    APPROVAL_REQUIRED = "approval_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    HEARTBEAT = "heartbeat"


class SessionCreateRequest(ApiContractModel):
    """Optional scope attached to a newly created conversation session."""

    teacher_id: Optional[ApiIdentifier] = None
    class_id: Optional[ApiIdentifier] = None


class SessionCreateResponse(ApiContractModel):
    session_id: ApiIdentifier
    thread_id: ApiIdentifier
    status: SessionStatus = SessionStatus.ACTIVE


class MessageCreateRequest(ApiContractModel):
    """A teacher message submitted within an existing session.

    ``request_id`` is optional for ordinary callers. A caller that needs safe
    retries may supply its own stable idempotency key; otherwise the future API
    runtime will generate one.
    """

    message: ApiMessage
    request_id: Optional[ApiIdentifier] = None


class MessageAcceptedResponse(ApiContractModel):
    session_id: ApiIdentifier
    request_id: ApiIdentifier
    status: RunStatus = RunStatus.ACCEPTED


class DraftResponse(ApiContractModel):
    """A completed or approval-pending draft exposed by the draft endpoint."""

    session_id: ApiIdentifier
    request_id: ApiIdentifier
    status: RunStatus
    draft: Draft
    approval: Approval
    citations: List[Citation] = Field(default_factory=list)


class ApprovalSubmitRequest(ApprovalResumeCommand):
    """HTTP form of the existing validated LangGraph resume command."""

    model_config = ConfigDict(extra="forbid")


class ApprovalSubmitResponse(ApiContractModel):
    session_id: ApiIdentifier
    request_id: ApiIdentifier
    decision: ApprovalDecision
    status: RunStatus


class CancelRunResponse(ApiContractModel):
    session_id: ApiIdentifier
    request_id: ApiIdentifier
    status: RunStatus = RunStatus.CANCELLED


class StreamEvent(ApiContractModel):
    """Payload serialized into one Server-Sent Event frame."""

    event_id: ApiIdentifier
    event: StreamEventType
    sequence: int = Field(ge=0)
    session_id: ApiIdentifier
    request_id: Optional[ApiIdentifier] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class ApiErrorDetail(ApiContractModel):
    code: ApiIdentifier
    message: ApiMessage
    recoverable: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class ApiErrorResponse(ApiContractModel):
    error: ApiErrorDetail
    request_id: Optional[ApiIdentifier] = None
