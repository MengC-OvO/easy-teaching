"""Tenant-isolated local knowledge indexes for approved centre documents."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import chromadb

from app.config import settings
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
from app.services.local_embedding_provider import (
    LocalSentenceTransformerEmbeddingProvider,
)
from app.services.vector_store import ChromaVectorStore


RRF_K = 60


class ScopedEmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    async def embed_texts_async(
        self,
        texts: List[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> Any: ...

    async def embed_text_async(
        self,
        text: str,
        *,
        task_type: str = "RETRIEVAL_QUERY",
    ) -> List[float]: ...


class ScopedVectorStore(Protocol):
    def upsert_chunks(
        self,
        chunks: List[KnowledgeChunk],
        embeddings: List[List[float]],
    ) -> int: ...

    def query(
        self,
        query_embedding: List[float],
        *,
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedKnowledgeChunk]: ...


class ScopedKnowledgeStore:
    """Build and query one local FTS index per trusted teacher/class scope."""

    def __init__(
        self,
        *,
        root: Path | str,
        file_store: LocalUploadedFileStore,
        document_reader: UploadedDocumentReader | None = None,
        ingestion_service: KnowledgeIngestionService | None = None,
        embedding_provider: ScopedEmbeddingProvider | None = None,
        vector_store_factory: Callable[[Path], ScopedVectorStore] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_store = file_store
        self.document_reader = document_reader or UploadedDocumentReader()
        self.ingestion_service = ingestion_service or KnowledgeIngestionService()
        self.embedding_provider = (
            embedding_provider
            or LocalSentenceTransformerEmbeddingProvider(
                model_name=settings.scoped_embedding_model_name,
                dimension=settings.scoped_embedding_dimension,
            )
        )
        self.vector_store_factory = vector_store_factory or self._default_vector_store
        self._vector_stores: Dict[Path, ScopedVectorStore] = {}
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

        embeddings = await self.embedding_provider.embed_texts_async(
            [chunk.retrieval_text for chunk in result.chunks],
            task_type="RETRIEVAL_DOCUMENT",
        )
        vectors = getattr(embeddings, "vectors", embeddings)
        if not isinstance(vectors, list) or len(vectors) != len(result.chunks):
            raise ValueError("Local embedding provider returned an invalid batch")

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
                self._vector_store(directory).upsert_chunks,
                result.chunks,
                vectors,
            )
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
            "index_mode": "tenant_local_hybrid",
            "content_hash": record.sha256,
        }

    async def search(
        self,
        *,
        query: str,
        top_k: int,
        teacher_id: str,
        class_id: str,
        return_candidate_pool: bool = False,
    ) -> List[RetrievedKnowledgeChunk]:
        directory = self._scope_directory(teacher_id, class_id, create=False)
        index_path = directory / "knowledge_fts.sqlite3"
        vector_path = directory / "chroma"
        if not index_path.is_file() and not vector_path.is_dir():
            return []
        candidate_top_k = min(20, max(top_k, top_k * 4))
        filters = RetrievalFilters(source_types=[KnowledgeSourceType.CENTRE])

        async def dense_search() -> List[RetrievedKnowledgeChunk]:
            if not vector_path.is_dir():
                return []
            embedding = self.embedding_provider.embed_text_async(
                query,
                task_type="RETRIEVAL_QUERY",
            )
            if inspect.isawaitable(embedding):
                embedding = await embedding
            return await asyncio.to_thread(
                self._vector_store(directory).query,
                embedding,
                top_k=candidate_top_k,
                where={"source_type": KnowledgeSourceType.CENTRE.value},
            )

        async def bm25_search() -> List[RetrievedKnowledgeChunk]:
            if not index_path.is_file():
                return []
            return await asyncio.to_thread(
                SQLiteFTS5KnowledgeIndex(index_path).search,
                query,
                top_k=candidate_top_k,
                filters=filters,
            )

        dense_chunks, bm25_chunks = await asyncio.gather(
            dense_search(),
            bm25_search(),
        )
        ranked = self._rank_fusion(dense_chunks, bm25_chunks)
        return ranked if return_candidate_pool else ranked[:top_k]

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

    def _vector_store(self, directory: Path) -> ScopedVectorStore:
        store = self._vector_stores.get(directory)
        if store is None:
            store = self.vector_store_factory(directory)
            self._vector_stores[directory] = store
        return store

    def _default_vector_store(self, directory: Path) -> ScopedVectorStore:
        scoped_settings = settings.model_copy(
            update={
                "embedding_model_name": self.embedding_provider.model_name,
                "embedding_dimension": self.embedding_provider.dimension,
            }
        )
        client = chromadb.PersistentClient(path=str(directory / "chroma"))
        return ChromaVectorStore(provider_settings=scoped_settings, client=client)

    @staticmethod
    def _rank_fusion(
        dense_chunks: List[RetrievedKnowledgeChunk],
        bm25_chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        chunks: Dict[str, RetrievedKnowledgeChunk] = {}
        scores: Dict[str, float] = {}
        for rank, item in enumerate(dense_chunks, start=1):
            chunk = item.model_copy(deep=True)
            chunk.dense_rank = rank
            chunks.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                0.60 / (RRF_K + rank)
            )
        for rank, item in enumerate(bm25_chunks, start=1):
            chunk = chunks.setdefault(item.chunk_id, item.model_copy(deep=True))
            chunk.bm25_score = item.bm25_score
            chunk.bm25_rank = item.bm25_rank or rank
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + (
                0.40 / (RRF_K + rank)
            )

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        result: List[RetrievedKnowledgeChunk] = []
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            chunk = chunks[chunk_id]
            chunk.fusion_score = scores[chunk_id]
            chunk.fusion_rank = rank
            chunk.final_rank = rank
            chunk.metadata = {
                **chunk.metadata,
                "scoped_hybrid_score": f"{scores[chunk_id]:.6f}",
            }
            result.append(chunk)
        return result

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
