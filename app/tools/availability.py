"""State-based Tool availability for the Main ReAct loop.

Local tools use lifecycle and call-budget checks. Dynamic Drive tools also use
conservative, explicit read-intent rules; ambiguous requests retain candidates.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping
import json
from app.config import settings

from app.schemas import CapabilityObservation, ObservationStatus
from app.tools.definition import ToolDefinition, ToolPermission


MAX_RETRIEVAL_ATTEMPTS = 2
RETRIEVAL_TOOLS = {
    "retrieve_knowledge",
    "query_records",
    "read_uploaded_document",
    "search_official_web",
}
ONE_SHOT_READ_TOOLS = {
    "get_class_context",
    "get_daily_context",
}


def available_tools_for_state(
    tools: Iterable[ToolDefinition],
    *,
    observations: Mapping[str, CapabilityObservation],
    tool_attempt_counts: Mapping[str, int],
    user_message: str = "",
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
        if not tool.model_visible:
            continue
        if tool.name.startswith("drive__") and user_message:
            message = user_message.casefold()
            drive_context = any(item.capability_name.startswith("drive") for item in observations.values())
            explicit_drive = any(word in message for word in ("drive", "google", "upload", "cloud", "上传", "云盘", "云端", "谷歌", "文件"))
            # Only narrow when intent is explicit. Ambiguous references keep
            # the budget-permitting catalog rather than silently hiding tools.
            wants_read = any(word in message for word in ("search", "find", "read", "list", "查询", "搜索", "读取", "列出"))
            wants_write = any(word in message for word in ("upload", "save", "create", "上传", "保存", "创建"))
            if (drive_context or explicit_drive) and wants_read and not wants_write and tool.permission is ToolPermission.REQUIRE_APPROVAL:
                continue
        if tool.name == "read_observation" and not any(item.body_ref for item in observations.values()):
            continue
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
    if sum(len(json.dumps(tool.model_spec(), ensure_ascii=False)) for tool in selected) > settings.tool_schema_max_chars:
        raise ValueError("Selected tool schemas exceed budget; narrow the task or configured catalog")
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
        if observation.is_partial:
            # A preview cannot establish that the complete issue set stalled.
            return False
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
