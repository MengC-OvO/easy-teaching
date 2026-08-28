"""Tenant-isolated local knowledge indexes for approved centre documents."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any, List

from app.schemas import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSourceType,
    RetrievalFilters,
    RetrievedKnowledgeChunk,
)
from app.services.document_reader import UploadedDocumentReader
from app.services.file_assets import LocalUploadedFileStore
from app.services.knowledge_ingestion import KnowledgeIngestionService
from app.services.lexical_index import SQLiteFTS5KnowledgeIndex


class ScopedKnowledgeStore:
    """Build and query one local FTS index per trusted teacher/class scope."""

    def __init__(
        self,
        *,
        root: Path | str,
        file_store: LocalUploadedFileStore,
        document_reader: UploadedDocumentReader | None = None,
        ingestion_service: KnowledgeIngestionService | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_store = file_store
        self.document_reader = document_reader or UploadedDocumentReader()
        self.ingestion_service = ingestion_service or KnowledgeIngestionService()
        self._lock = asyncio.Lock()

    async def ingest(
        self,
        *,
        file_id: str,
        title: str,
        teacher_id: str,
        class_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        record = self.file_store.get_authorized(
            file_id,
            teacher_id=teacher_id,
            class_id=class_id,
            session_id=session_id,
            required_category="document",
        )
        scope_key = self._scope_key(teacher_id, class_id)
        source_id = f"centre-{scope_key}-{file_id}"
        document = KnowledgeDocument(
            source_id=source_id,
            source_type=KnowledgeSourceType.CENTRE,
            title=title.strip() or record.original_name,
            version=record.sha256[:12],
            uri=f"upload://{file_id}",
        )
        blocks = await asyncio.to_thread(self.document_reader.to_parsed_blocks, record)
        result = self.ingestion_service.ingest_blocks(document, blocks)
        if not result.chunks:
            raise ValueError("The uploaded document contains no indexable text")

        async with self._lock:
            directory = self._scope_directory(teacher_id, class_id)
            chunks_path = directory / "chunks.jsonl"
            existing = [
                chunk
                for chunk in self._read_chunks(chunks_path)
                if chunk.document.source_id != source_id
            ]
            chunks = list({item.chunk_id: item for item in [*existing, *result.chunks]}.values())
            await asyncio.to_thread(
                SQLiteFTS5KnowledgeIndex.build,
                directory / "knowledge_fts.sqlite3",
                chunks,
                source_digest=record.sha256,
            )
            self._write_chunks(chunks_path, chunks)
        return {
            "source_id": source_id,
            "title": document.title,
            "chunk_count": result.chunk_count,
            "source_type": KnowledgeSourceType.CENTRE.value,
            "index_mode": "tenant_local_bm25",
            "content_hash": record.sha256,
        }

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        teacher_id: str,
        class_id: str,
    ) -> List[RetrievedKnowledgeChunk]:
        index_path = self._scope_directory(teacher_id, class_id, create=False) / "knowledge_fts.sqlite3"
        if not index_path.is_file():
            return []
        index = SQLiteFTS5KnowledgeIndex(index_path)
        return await asyncio.to_thread(
            index.search,
            query,
            top_k=top_k,
            filters=RetrievalFilters(source_types=[KnowledgeSourceType.CENTRE]),
        )

    def preview(self, *, file_id: str, teacher_id: str, class_id: str, session_id: str) -> dict[str, Any]:
        record = self.file_store.get_authorized(
            file_id,
            teacher_id=teacher_id,
            class_id=class_id,
            session_id=session_id,
            required_category="document",
        )
        result = self.document_reader.read(record, max_chars=1_000)
        return {
            "file_id": file_id,
            "filename": record.original_name,
            "size_bytes": record.size_bytes,
            "content_hash": record.sha256,
            "preview": "\n".join(section.text for section in result.sections)[:1_000],
        }

    def _scope_directory(self, teacher_id: str, class_id: str, *, create: bool = True) -> Path:
        directory = (self.root / self._scope_key(teacher_id, class_id)).resolve()
        if directory.parent != self.root:
            raise ValueError("Invalid knowledge scope")
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _scope_key(teacher_id: str, class_id: str) -> str:
        if not teacher_id or not class_id:
            raise PermissionError("Trusted teacher and class scope is required")
        return hashlib.sha256(f"{teacher_id}\0{class_id}".encode()).hexdigest()[:24]

    @staticmethod
    def _read_chunks(path: Path) -> List[KnowledgeChunk]:
        if not path.is_file():
            return []
        return [KnowledgeChunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _write_chunks(path: Path, chunks: List[KnowledgeChunk]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("".join(chunk.model_dump_json() + "\n" for chunk in chunks), encoding="utf-8")
        os.replace(temporary, path)
