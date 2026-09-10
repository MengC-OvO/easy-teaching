"""Bounded reads of existing immutable snapshots, never replays original tools."""
import json
from pydantic import BaseModel, Field, ConfigDict
from app.schemas import RiskLevel
from app.services.observation_results import fingerprint
from app.tools.definition import ToolDefinition, ToolCategory, ToolDomain, ToolPermission, ToolResult


class ReadObservationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body_ref: str = Field(min_length=64, max_length=64)
    result_key: str | None = None
    section: str | None = Field(default=None, max_length=200)
    cursor: int = Field(default=0, ge=0)
    limit: int = Field(default=2000, ge=1, le=4000)


class ReadObservationOutput(BaseModel):
    body_ref: str
    content_hash: str
    result_key: str
    source_request_id: str
    section: str | None = None
    text: str
    format: str
    start: int
    end: int
    next_cursor: int | None
    has_more: bool
    is_partial: bool


def build_read_observation_tool(store):
    async def read(data, context):
        original = await store.read_tool_result_snapshot(body_ref=data.body_ref,
            session_id=context.session_id, teacher_id=context.teacher_id, class_id=context.class_id)
        if fingerprint(original["body"]) != original["content_hash"]:
            raise ValueError("Snapshot integrity mismatch")
        if data.result_key is not None and original["result_key"] != data.result_key:
            raise ValueError("Snapshot does not match the requested result")
        value = original["body"]
        for part in (data.section or "").split(".") if data.section else []:
            if not isinstance(value, dict) or part not in value:
                raise ValueError("Requested section does not exist")
            value = value[part]
        is_text = isinstance(value, str)
        text = value if is_text else json.dumps(value, ensure_ascii=False, sort_keys=True)
        if data.cursor > len(text):
            raise ValueError("Cursor exceeds this section")
        end = min(len(text), data.cursor + data.limit)
        return ToolResult.ok(risk_level=RiskLevel.L0_READ_ONLY, data={
            "body_ref": data.body_ref, "content_hash": original["content_hash"],
            "result_key": original["result_key"], "source_request_id": original["request_id"],
            "section": data.section, "text": text[data.cursor:end],
            "format": "text_fragment" if is_text else "serialized_json_text_fragment",
            "start": data.cursor, "end": end, "next_cursor": end if end < len(text) else None,
            "has_more": end < len(text), "is_partial": data.cursor > 0 or end < len(text) or bool(data.section),
        })
    return ToolDefinition(name="read_observation", description="Read a bounded fragment of a previous tool result using its body_ref. This is the immutable old result, not a fresh query. section selects a dotted object field; cursor/limit are character offsets. JSON fragments are text, not complete JSON objects.",
        category=ToolCategory.FILE, domain=ToolDomain.INTERNAL, input_model=ReadObservationInput,
        output_model=ReadObservationOutput, risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE, async_runtime_handler=read,
        parallel_safe=True, max_identical_calls_per_run=1)
