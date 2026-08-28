"""Optional local speech-to-text provider for scoped teacher voice notes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str


class TranscriptionResult(BaseModel):
    text: str
    language: str
    language_probability: Optional[float] = Field(default=None, ge=0, le=1)
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    segments: List[TranscriptSegment]


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model_name: str = "small.en",
        device: str = "auto",
        compute_type: str = "int8_float16",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None

    async def transcribe(
        self,
        path: Path,
        *,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        return await asyncio.to_thread(self._transcribe_sync, path, language)

    def _transcribe_sync(
        self,
        path: Path,
        language: Optional[str],
    ) -> TranscriptionResult:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise RuntimeError(
                    "Voice transcription requires requirements-transcription.txt"
                ) from error
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        raw_segments, info = self._model.transcribe(
            str(path),
            language=language,
            vad_filter=True,
            beam_size=5,
        )
        segments = [
            TranscriptSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=str(segment.text).strip(),
            )
            for segment in raw_segments
            if str(segment.text).strip()
        ]
        return TranscriptionResult(
            text=" ".join(item.text for item in segments).strip(),
            language=str(getattr(info, "language", language or "unknown")),
            language_probability=(
                float(info.language_probability)
                if getattr(info, "language_probability", None) is not None
                else None
            ),
            duration_seconds=(
                float(info.duration) if getattr(info, "duration", None) is not None else None
            ),
            segments=segments,
        )

