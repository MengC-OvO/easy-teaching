"""State-based Tool availability for the Main ReAct loop.

This module intentionally does not classify user text. Main receives every
registered, permitted Tool schema until an observed lifecycle fact makes a Tool
unavailable for the current run.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping

from app.schemas import CapabilityObservation, ObservationStatus
from app.tools.definition import ToolDefinition, ToolPermission


MAX_RETRIEVAL_ATTEMPTS = 2
RETRIEVAL_TOOLS = {"retrieve_knowledge", "query_records", "search_google_drive"}
ONE_SHOT_READ_TOOLS = {
    "get_class_context",
    "get_daily_context",
}


def available_tools_for_state(
    tools: Iterable[ToolDefinition],
    *,
    observations: Mapping[str, CapabilityObservation],
    tool_attempt_counts: Mapping[str, int],
) -> List[ToolDefinition]:
    """Return Tools still useful after deterministic execution-state checks."""

    completed = {
        observation.capability_name
        for observation in observations.values()
        if observation.status is ObservationStatus.COMPLETED
    }
    completed_counts: dict[str, int] = {}
    for observation in observations.values():
        if observation.status is ObservationStatus.COMPLETED:
            completed_counts[observation.capability_name] = (
                completed_counts.get(observation.capability_name, 0) + 1
            )
    selected: List[ToolDefinition] = []
    for tool in tools:
        if tool.permission is ToolPermission.FORBIDDEN:
            continue
        if tool.name in ONE_SHOT_READ_TOOLS and tool.name in completed:
            continue
        if (
            tool.max_successful_calls_per_run is not None
            and completed_counts.get(tool.name, 0)
            >= tool.max_successful_calls_per_run
        ):
            continue
        if tool.name == "check_activity_safety" and _safety_progress_stalled(
            observations
        ):
            continue
        if (
            tool.name in RETRIEVAL_TOOLS
            and tool_attempt_counts.get(tool.name, 0) >= MAX_RETRIEVAL_ATTEMPTS
        ):
            continue
        selected.append(tool)
    return selected


def _safety_progress_stalled(
    observations: Mapping[str, CapabilityObservation],
) -> bool:
    """Stop automatic revision churn when the latest check reduced no risks.

    There is no arbitrary successful-call ceiling. A sequence may continue while
    issue codes strictly decrease, which is finite and evidence-based. A later
    teacher message starts a fresh run and can check another revision normally.
    """

    issue_sets = []
    for observation in observations.values():
        if (
            observation.capability_name != "check_activity_safety"
            or observation.status is not ObservationStatus.COMPLETED
        ):
            continue
        issues = observation.data.get("issues")
        if not isinstance(issues, list):
            continue
        issue_sets.append(
            {
                str(item.get("code"))
                for item in issues
                if isinstance(item, dict) and item.get("code")
            }
        )
    if len(issue_sets) < 2:
        return False
    previous, current = issue_sets[-2:]
    return not current < previous
