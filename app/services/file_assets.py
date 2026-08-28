"""Scoped local storage for teacher-uploaded documents and audio."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"}
ALLOWED_UPLOAD_EXTENSIONS = DOCUMENT_EXTENSIONS | AUDIO_EXTENSIONS
_FILE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class UploadedFileRecord(BaseModel):
    file_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    original_name: str
    stored_path: str
    content_type: str
    size_bytes: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64)
    category: str
    teacher_id: str
    class_id: str
    session_id: str
    created_at: str


class LocalUploadedFileStore:
    """Persist opaque file IDs under one configured root; never accept model paths."""

    def __init__(self, root: Path | str, *, max_bytes: int = 15 * 1024 * 1024) -> None:
        self.root = Path(root).resolve()
        self.max_bytes = max_bytes
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(
        self,
        *,
        filename: str,
        content_type: Optional[str],
        content: bytes,
        teacher_id: str,
        class_id: str,
        session_id: str,
    ) -> UploadedFileRecord:
        safe_name = self._safe_filename(filename)
        extension = Path(safe_name).suffix.casefold()
        if extension not in ALLOWED_UPLOAD_EXTENSIONS:
            raise ValueError(f"Unsupported upload extension: {extension or '[none]'}")
        if not content:
            raise ValueError("Uploaded file is empty")
        if len(content) > self.max_bytes:
            raise ValueError(f"Uploaded file exceeds {self.max_bytes} bytes")
        if not teacher_id or not class_id or not session_id:
            raise ValueError("Upload requires trusted teacher, class and session scope")

        file_id = uuid4().hex
        directory = self.root / file_id
        directory.mkdir(parents=False, exist_ok=False)
        target = directory / safe_name
        temporary = directory / f".{safe_name}.tmp"
        try:
            temporary.write_bytes(content)
            os.replace(temporary, target)
            record = UploadedFileRecord(
                file_id=file_id,
                original_name=safe_name,
                stored_path=str(target),
                content_type=(content_type or "application/octet-stream")[:200],
                size_bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),
                category="audio" if extension in AUDIO_EXTENSIONS else "document",
                teacher_id=teacher_id,
                class_id=class_id,
                session_id=session_id,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            metadata_temp = directory / ".metadata.json.tmp"
            metadata_temp.write_text(
                record.model_dump_json(indent=2), encoding="utf-8"
            )
            os.replace(metadata_temp, directory / "metadata.json")
            return record
        except Exception:
            temporary.unlink(missing_ok=True)
            if target.exists():
                target.unlink(missing_ok=True)
            for item in directory.glob("*"):
                item.unlink(missing_ok=True)
            directory.rmdir()
            raise

    def get_authorized(
        self,
        file_id: str,
        *,
        teacher_id: str,
        class_id: str,
        session_id: Optional[str] = None,
        required_category: Optional[str] = None,
    ) -> UploadedFileRecord:
        record = self.get(file_id)
        if record.teacher_id != teacher_id or record.class_id != class_id:
            raise PermissionError("Uploaded file is outside the trusted teacher/class scope")
        if session_id is not None and record.session_id != session_id:
            raise PermissionError("Uploaded file belongs to another session")
        if required_category is not None and record.category != required_category:
            raise ValueError(f"Uploaded file must be a {required_category}")
        return record

    def get(self, file_id: str) -> UploadedFileRecord:
        if not _FILE_ID_PATTERN.fullmatch(file_id):
            raise ValueError("Invalid uploaded file ID")
        directory = (self.root / file_id).resolve()
        if directory.parent != self.root:
            raise ValueError("Invalid uploaded file path")
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("Uploaded file does not exist")
        record = UploadedFileRecord.model_validate_json(
            metadata_path.read_text(encoding="utf-8")
        )
        stored_path = Path(record.stored_path).resolve()
        if stored_path.parent != directory or not stored_path.is_file():
            raise FileNotFoundError("Uploaded file content is unavailable")
        return record

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename or "").name.strip()
        name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)[:160]
        if not name or name in {".", ".."}:
            raise ValueError("Uploaded filename is invalid")
        return name

