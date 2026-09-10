"""Shared ownership and terminal-state checks for checkpoint recovery."""
from typing import Any, Mapping

from app.schemas import GraphState, RunStatus, WorkflowStatus


def checkpoint_belongs_to_run(
    values: Mapping[str, Any], *, request_id: str, session_id: str
) -> bool:
    return (
        values.get("request_id") == request_id
        and values.get("session_id") == session_id
    )


def completed_snapshot_status(state: GraphState) -> RunStatus:
    """Only a terminal graph state can be finalized without invoking the graph."""
    statuses = {
        WorkflowStatus.COMPLETED: RunStatus.COMPLETED,
        WorkflowStatus.FAILED: RunStatus.FAILED,
        WorkflowStatus.WAITING_FOR_APPROVAL: RunStatus.WAITING_FOR_APPROVAL,
    }
    if state.workflow_status not in statuses:
        raise ValueError("Checkpoint has no next node but the run is not terminal")
    return statuses[state.workflow_status]
