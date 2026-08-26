"""Validated command supplied when a teacher resumes an approval pause.

This is intentionally separate from ``ApprovalStatus``.  A decision is the
teacher's requested action; the status is the state recorded by the Agent graph
after it has checked and applied that action.
"""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, StringConstraints


ApprovalRequestId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]


class ApprovalDecision(str, Enum):
    """The only actions a teacher may take on a waiting draft."""

    APPROVE = "approve"
    REJECT = "reject"


class ApprovalResumeCommand(BaseModel):
    """A generic, validated instruction for an approval-gated Agent action.

    ``request_id`` identifies the draft that the human intended to act on.
    Before resuming, the API compares it with the paused graph
    state before changing approval status or saving anything.
    """

    request_id: ApprovalRequestId
    decision: ApprovalDecision
