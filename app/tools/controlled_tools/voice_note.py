"""Scoped local voice-note transcription Tool definition."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.services.file_assets import LocalUploadedFileStore
from app.tools.controlled_tools.upload_scope import require_upload_scope
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


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


def build_transcribe_voice_note_tool(
    file_store: LocalUploadedFileStore,
    transcriber: Any,
) -> ToolDefinition:
    async def run(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = require_upload_scope(context)
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
        result = await transcriber.transcribe(
            Path(record.stored_path),
            language=data.language,
        )
        payload = result.model_dump(mode="json")
        payload.update({"file_id": data.file_id, "filename": record.original_name})
        return ToolResult.ok(data=payload, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="transcribe_voice_note",
        description=(
            "Transcribe an audio note uploaded in this conversation with the "
            "configured local Whisper model. It does not save an observation "
            "automatically."
        ),
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
