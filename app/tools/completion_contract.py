"""Deterministic completion contracts for explicit controlled operations.

The Main model may choose how to fulfil a request, but it is not trusted to
declare whether an explicit save/export/upload request has been completed.
Controlled tools register their own high-confidence language aliases here via
ToolDefinition metadata, so future tools use the same policy without graph
special-cases.
"""

from __future__ import annotations

import re
from typing import Iterable, List

from app.tools.definition import ToolDefinition, ToolPermission


_NEGATION = re.compile(
    r"(?:不要|不用|无需|别|不需要|do\s+not|don't|dont|without)\s*$",
    re.IGNORECASE,
)


def resolve_required_controlled_tools(
    message: str,
    tools: Iterable[ToolDefinition],
) -> List[str]:
    """Resolve only explicit, unnegated controlled-operation requests."""

    normalized = " ".join(message.casefold().split())
    required: List[str] = []
    for tool in tools:
        if (
            tool.permission is not ToolPermission.REQUIRE_APPROVAL
            and tool.permission_resolver is None
        ):
            continue
        if any(_contains_unnegated_alias(normalized, alias) for alias in tool.completion_aliases):
            required.append(tool.name)
    return list(dict.fromkeys(required))


def _contains_unnegated_alias(message: str, alias: str) -> bool:
    target = " ".join(alias.casefold().split())
    if not target:
        return False
    start = message.find(target)
    while start >= 0:
        prefix = message[max(0, start - 16) : start]
        if not _NEGATION.search(prefix):
            return True
        start = message.find(target, start + len(target))
    return False
