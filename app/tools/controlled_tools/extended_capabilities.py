"""Scoped file, web, analytics and media capabilities."""

from __future__ import annotations

import asyncio
import inspect
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas import RiskLevel
from app.services.document_reader import UploadedDocumentReader
from app.services.file_assets import LocalUploadedFileStore
from app.tools.definition import (
    PreparedToolAction,
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolErrorCode,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


def _require_scope(context: ToolExecutionContext, *, session: bool = False) -> Optional[ToolResult]:
    if context.teacher_id and context.class_id and (context.session_id or not session):
        return None
    return ToolResult.fail(
        code=ToolErrorCode.PERMISSION_DENIED,
        message="This capability requires trusted teacher, class and session scope.",
        risk_level=RiskLevel.L3_FORBIDDEN,
        recoverable=False,
    )


class ReadUploadedDocumentInput(BaseModel):
    file_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    max_chars: int = Field(default=12_000, ge=500, le=20_000)


class ReadUploadedDocumentOutput(BaseModel):
    file_id: str
    filename: str
    content_type: str
    sections: List[Dict[str, Any]]
    extracted_chars: int
    truncated: bool


def build_read_uploaded_document_tool(file_store: LocalUploadedFileStore, reader: UploadedDocumentReader) -> ToolDefinition:
    async def run(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = _require_scope(context, session=True)
        if denied:
            return denied
        data = ReadUploadedDocumentInput.model_validate(input_data)
        record = file_store.get_authorized(
            data.file_id,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            session_id=context.session_id,
            required_category="document",
        )
        result = await asyncio.to_thread(reader.read, record, max_chars=data.max_chars)
        return ToolResult.ok(data=result.model_dump(mode="json"), risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="read_uploaded_document",
        description="Read text from a document the teacher uploaded in this conversation. Use the opaque file_id returned by the upload API; never invent paths.",
        category=ToolCategory.FILE,
        input_model=ReadUploadedDocumentInput,
        output_model=ReadUploadedDocumentOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        async_runtime_handler=run,
    )


class IngestUploadedDocumentInput(BaseModel):
    file_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    title: str = Field(min_length=2, max_length=300)


class IngestUploadedDocumentOutput(BaseModel):
    source_id: str
    title: str
    chunk_count: int
    source_type: Literal["centre"]
    index_mode: Literal["tenant_local_bm25"]
    content_hash: str


def build_ingest_uploaded_document_tool(scoped_knowledge: Any) -> ToolDefinition:
    async def prepare(input_data: BaseModel, context: ToolExecutionContext) -> PreparedToolAction:
        data = IngestUploadedDocumentInput.model_validate(input_data)
        if _require_scope(context, session=True):
            raise ValueError("Trusted upload scope is required")
        preview = scoped_knowledge.preview(
            file_id=data.file_id,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            session_id=context.session_id,
        )
        return PreparedToolAction(arguments=data.model_dump(mode="json"), preview={**preview, "title": data.title, "index_mode": "tenant_local_bm25"})

    async def run(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = _require_scope(context, session=True)
        if denied:
            return denied
        data = IngestUploadedDocumentInput.model_validate(input_data)
        result = await scoped_knowledge.ingest(
            file_id=data.file_id,
            title=data.title,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            session_id=context.session_id,
        )
        return ToolResult.ok(data=result, risk_level=RiskLevel.L2_CONTROLLED_WRITE)

    return ToolDefinition(
        name="ingest_uploaded_document",
        description="After teacher approval, add an uploaded centre document to that teacher/class's isolated local BM25 knowledge index. Reading alone does not index it.",
        category=ToolCategory.FILE,
        input_model=IngestUploadedDocumentInput,
        output_model=IngestUploadedDocumentOutput,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        completion_aliases=("加入知识库", "添加到知识库", "add to the knowledge base", "index this document"),
        domain=ToolDomain.LOCAL,
        parallel_safe=False,
        async_runtime_handler=run,
        approval_preparation_handler=prepare,
    )


class OfficialWebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    domains: List[str] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=5, ge=1, le=10)


class OfficialWebSearchOutput(BaseModel):
    query: str
    results: List[Dict[str, str]]
    returned_count: int


def build_official_web_search_tool(client: Any) -> ToolDefinition:
    async def run(input_data: BaseModel) -> ToolResult:
        data = OfficialWebSearchInput.model_validate(input_data)
        response = await client.search(data.query, domains=data.domains or None, top_k=data.top_k)
        payload = response.model_dump(mode="json")
        payload["returned_count"] = len(payload["results"])
        return ToolResult.ok(data=payload, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="search_official_web",
        description="Search current Australian education guidance on configured government and ACECQA domains only. Use for recent official facts not covered by the local RAG corpus.",
        category=ToolCategory.WEB,
        input_model=OfficialWebSearchInput,
        output_model=OfficialWebSearchOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.EXTERNAL,
        parallel_safe=True,
        timeout_seconds=15,
        async_handler=run,
    )


class AnalyseLearningRecordsInput(BaseModel):
    record_type: Literal["observation", "educational_record", "all"] = "all"
    group_by: Literal["record_type", "status", "month"] = "record_type"
    child_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    status: Optional[Literal["draft", "final", "archived"]] = None
    limit: int = Field(default=50, ge=1, le=50)

    @model_validator(mode="after")
    def valid_dates(self) -> "AnalyseLearningRecordsInput":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class AnalyseLearningRecordsOutput(BaseModel):
    total_records: int
    group_by: str
    groups: List[Dict[str, Any]]
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    truncated: bool


def build_analyse_learning_records_tool(store: Any) -> ToolDefinition:
    async def run(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = _require_scope(context)
        if denied:
            return denied
        data = AnalyseLearningRecordsInput.model_validate(input_data)
        records = store.query_records(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            record_type=data.record_type,
            child_id=data.child_id,
            date_from=data.date_from,
            date_to=data.date_to,
            status=data.status,
            search_text=None,
            limit=data.limit,
        )
        if inspect.isawaitable(records):
            records = await records
        keys = [_analytics_key(record, data.group_by) for record in records]
        counts = Counter(keys)
        output = AnalyseLearningRecordsOutput(
            total_records=len(records),
            group_by=data.group_by,
            groups=[{"key": key, "count": count} for key, count in sorted(counts.items())],
            date_from=data.date_from.isoformat() if data.date_from else None,
            date_to=data.date_to.isoformat() if data.date_to else None,
            truncated=len(records) >= data.limit,
        )
        return ToolResult.ok(data=output.model_dump(mode="json"), risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="analyse_learning_records",
        description="Compute bounded counts and trends over authorised class observations and educational records. Use for summaries or distributions, not for reading record prose.",
        category=ToolCategory.ANALYTICS,
        input_model=AnalyseLearningRecordsInput,
        output_model=AnalyseLearningRecordsOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        async_runtime_handler=run,
    )


def _analytics_key(record: Dict[str, Any], group_by: str) -> str:
    if group_by == "month":
        value = str(record.get("observed_at") or record.get("created_at") or "")
        return value[:7] if len(value) >= 7 else "unknown"
    return str(record.get(group_by) or "unknown")


class TranscribeVoiceNoteInput(BaseModel):
    file_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    language: Optional[str] = Field(default=None, min_length=2, max_length=10)


class TranscribeVoiceNoteOutput(BaseModel):
    file_id: str
    filename: str
    text: str
    language: str
    language_probability: Optional[float] = None
    duration_seconds: Optional[float] = None
    segments: List[Dict[str, Any]]


def build_transcribe_voice_note_tool(file_store: LocalUploadedFileStore, transcriber: Any) -> ToolDefinition:
    async def run(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = _require_scope(context, session=True)
        if denied:
            return denied
        data = TranscribeVoiceNoteInput.model_validate(input_data)
        record = file_store.get_authorized(
            data.file_id,
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            session_id=context.session_id,
            required_category="audio",
        )
        result = await transcriber.transcribe(Path(record.stored_path), language=data.language)
        payload = result.model_dump(mode="json")
        payload.update({"file_id": data.file_id, "filename": record.original_name})
        return ToolResult.ok(data=payload, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="transcribe_voice_note",
        description="Transcribe an audio note uploaded in this conversation with the configured local Whisper model. It does not save an observation automatically.",
        category=ToolCategory.MEDIA,
        input_model=TranscribeVoiceNoteInput,
        output_model=TranscribeVoiceNoteOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=False,
        timeout_seconds=120,
        max_successful_calls_per_run=1,
        async_runtime_handler=run,
    )
