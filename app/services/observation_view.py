"""Bounded model-facing views of canonical capability observations.

The graph keeps complete observations for citations, approvals, debugging, and
checkpoint recovery.  Only the copy inserted into a model prompt is compacted.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

from app.schemas import CapabilityObservation


_MAX_GENERIC_STRING_CHARS = 600
_MAX_EVIDENCE_CONTENT_CHARS = 900
_MAX_LIST_ITEMS = 5
_DETAILED_HISTORY_ITEMS = 4
_MAX_DRAFT_ARTIFACT_CHARS = 20_000


def build_model_observation_view(
    observations: Mapping[str, CapabilityObservation],
) -> Dict[str, Dict[str, Any]]:
    """Return a small, evidence-safe view without mutating canonical state."""

    items = list(observations.items())
    detailed_keys = {key for key, _ in items[-_DETAILED_HISTORY_ITEMS:]}
    view: Dict[str, Dict[str, Any]] = {}
    for key, observation in items:
        base: Dict[str, Any] = {
            "result_key": observation.result_key,
            "capability_name": observation.capability_name,
            "source_kind": observation.source_kind.value,
            "status": observation.status.value,
        }
        if observation.error:
            base["error"] = _compact_generic(observation.error)
        if key in detailed_keys or observation.capability_name == "retrieve_knowledge":
            base["data"] = _compact_capability_data(
                observation.capability_name,
                observation.data,
            )
        else:
            base["summary"] = _historical_summary(observation)
        view[key] = base
    return view


def _compact_capability_data(name: str, data: Mapping[str, Any]) -> Dict[str, Any]:
    if name == "retrieve_knowledge":
        return _compact_knowledge(data)
    if name == "query_records":
        records = data.get("records")
        return {
            "returned_count": data.get("returned_count", 0),
            "search_text": data.get("search_text"),
            "records": [
                _compact_record(item)
                for item in (records if isinstance(records, list) else [])[:_MAX_LIST_ITEMS]
                if isinstance(item, dict)
            ],
        }
    if name == "read_uploaded_document":
        return {
            "file_id": data.get("file_id"),
            "filename": data.get("filename"),
            "sections": _compact_generic(data.get("sections", []), max_string_chars=1_200),
            "extracted_chars": data.get("extracted_chars", 0),
            "truncated": data.get("truncated", False),
        }
    if name == "search_official_web":
        return {
            "query": _truncate(data.get("query", ""), 500),
            "returned_count": data.get("returned_count", 0),
            "results": _compact_generic(data.get("results", []), max_string_chars=700),
        }
    if name == "analyse_learning_records":
        return {
            key: _compact_generic(data.get(key))
            for key in ("total_records", "group_by", "groups", "date_from", "date_to", "truncated")
        }
    if name == "transcribe_voice_note":
        return {
            "file_id": data.get("file_id"),
            "filename": data.get("filename"),
            "text": _truncate(data.get("text", ""), 4_000),
            "language": data.get("language"),
            "duration_seconds": data.get("duration_seconds"),
        }
    if name == "read_draft_artifact":
        content = str(data.get("content") or "")
        return {
            "source_request_id": data.get("source_request_id"),
            "title": data.get("title"),
            "content": _truncate(content, _MAX_DRAFT_ARTIFACT_CHARS),
            "content_chars": data.get("content_chars", len(content)),
            "created_at": data.get("created_at"),
            "status": data.get("status"),
        }
    if name == "check_activity_safety":
        checked_text = str(data.get("recovery_content") or "")
        return {
            "status": data.get("status"),
            "issues": _compact_generic(data.get("issues", [])),
            # Main calls are stateless between ReAct steps. Return the exact
            # inspected candidate so final fingerprint matching is achievable.
            "checked_activity_text": _truncate(
                checked_text,
                _MAX_DRAFT_ARTIFACT_CHARS,
            ),
            "content_fingerprint": data.get("content_fingerprint"),
        }
    if name in {"search_google_drive", "upload_export_to_google_drive"}:
        return {
            key: _compact_generic(value, max_string_chars=1_200)
            for key, value in data.items()
            if key in {"results_text", "result_text", "provider"}
        }
    if name.endswith("_worker"):
        return {
            "summary": _compact_generic(data.get("summary", ""), max_string_chars=2_000),
            "supporting_tool_count": len(data.get("tool_observations", [])),
        }
    return _compact_generic(data)


def _compact_knowledge(data: Mapping[str, Any]) -> Dict[str, Any]:
    evidence = data.get("evidence")
    compact_evidence = []
    strategy = str(data.get("strategy") or "standard").casefold()
    evidence_limit = 5 if strategy in {"deep", "enhanced"} else 3
    for item in (evidence if isinstance(evidence, list) else [])[:evidence_limit]:
        if not isinstance(item, dict):
            continue
        compact_evidence.append(
            {
                "evidence_id": item.get("evidence_id"),
                "content": _truncate(item.get("content", ""), _MAX_EVIDENCE_CONTENT_CHARS),
                "citation": _compact_generic(item.get("citation", {})),
                "final_rank": item.get("final_rank"),
            }
        )
    return {
        "query": _truncate(data.get("query", ""), 500),
        "strategy": data.get("strategy"),
        "knowledge_scope": data.get("knowledge_scope"),
        "returned_count": data.get("returned_count", len(compact_evidence)),
        "evidence": compact_evidence,
    }


def _compact_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    useful_keys = (
        "record_type",
        "observation_id",
        "record_id",
        "title",
        "observed_at",
        "setting",
        "objective_text",
        "educator_actions",
        "analysis",
        "curriculum_links",
        "next_steps",
        "status",
    )
    return {
        key: _compact_generic(record[key], max_string_chars=700)
        for key in useful_keys
        if key in record
    }


def _historical_summary(observation: CapabilityObservation) -> str:
    data = observation.data
    count = data.get("returned_count")
    suffix = f"; returned_count={count}" if isinstance(count, int) else ""
    return (
        f"{observation.capability_name} finished with status="
        f"{observation.status.value}{suffix}. Full result remains in canonical state."
    )


def _compact_generic(
    value: Any,
    *,
    depth: int = 0,
    max_string_chars: int = _MAX_GENERIC_STRING_CHARS,
) -> Any:
    if depth >= 4:
        return "[nested data omitted]"
    if isinstance(value, str):
        return _truncate(value, max_string_chars)
    if isinstance(value, dict):
        return {
            str(key): _compact_generic(
                nested,
                depth=depth + 1,
                max_string_chars=max_string_chars,
            )
            for key, nested in list(value.items())[:12]
        }
    if isinstance(value, list):
        return [
            _compact_generic(
                item,
                depth=depth + 1,
                max_string_chars=max_string_chars,
            )
            for item in value[:_MAX_LIST_ITEMS]
        ]
    return value


def _truncate(value: Any, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…[truncated]"
