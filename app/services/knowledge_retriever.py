import asyncio
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, Tuple

from app.schemas import (
    RetrievedKnowledgeChunk,
    RerankerMode,
    RetrievalFilters,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStats,
)
from app.config import settings
from app.services.embedding_provider import GeminiEmbeddingProvider
from app.services.lexical_index import SQLiteFTS5KnowledgeIndex
from app.services.vector_store import ChromaVectorStore


DEFAULT_CHUNKS_PATH = Path("data/knowledge/processed/chunks.jsonl")
RRF_K = 60


class QueryEmbeddingProvider(Protocol):
    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        ...

    async def embed_text_async(
        self, text: str, *, task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        ...


class KnowledgeVectorStore(Protocol):
    def query(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, object]] = None,
    ) -> List[RetrievedKnowledgeChunk]:
        ...


class KnowledgeLexicalIndex(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters,
    ) -> List[RetrievedKnowledgeChunk]:
        ...


class KnowledgeReranker(Protocol):
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = settings.reranker_model_name) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install project dependencies with `pip install -r requirements.txt`."
            ) from error
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = [float(score) for score in self.model.predict(pairs)]
        original_ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(chunks, 1)}
        scored_chunks = sorted(
            zip(scores, chunks), key=lambda item: item[0], reverse=True
        )
        cross_encoder_ranks = {
            chunk.chunk_id: rank
            for rank, (_, chunk) in enumerate(scored_chunks, start=1)
        }
        # Blend the trusted coarse rank with the semantic reranker rank. A pure
        # Cross-encoder sort can over-promote generally related policy passages
        # above the exact EYLF/NQS evidence that hybrid retrieval already found.
        scored_chunks.sort(
            key=lambda item: (
                2 / (60 + original_ranks[item[1].chunk_id])
                + 1 / (60 + cross_encoder_ranks[item[1].chunk_id])
            ),
            reverse=True,
        )

        reranked: List[RetrievedKnowledgeChunk] = []
        for score, chunk in scored_chunks:
            chunk.metadata = {
                **chunk.metadata,
                "cross_encoder_score": f"{score:.6f}",
                "cross_encoder_rank": str(cross_encoder_ranks[chunk.chunk_id]),
                "pre_reranker_rank": str(original_ranks[chunk.chunk_id]),
                "cross_encoder_model": self.model_name,
            }
            chunk.reranker_score = score
            chunk.reranker_rank = len(reranked) + 1
            reranked.append(chunk)
        return reranked


class KnowledgeRetriever:
    def __init__(
        self,
        *,
        embedding_provider: Optional[QueryEmbeddingProvider] = None,
        vector_store: Optional[KnowledgeVectorStore] = None,
        lexical_index: Optional[KnowledgeLexicalIndex] = None,
        cross_encoder_reranker: Optional[KnowledgeReranker] = None,
        candidate_multiplier: int = 4,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.lexical_index = lexical_index
        self.cross_encoder_reranker = cross_encoder_reranker
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        candidate_top_k = self._candidate_top_k(request.top_k)
        dense_chunks = self._dense_search(request, candidate_top_k)
        bm25_chunks = self._bm25_search(request, candidate_top_k)
        chunks = self._combine_results(request, dense_chunks, bm25_chunks)
        deduplicated_chunks = self._deduplicate(chunks)
        reranked_chunks = self._rerank(
            request.query,
            deduplicated_chunks,
            mode=request.reranker,
        )
        returned_chunks = self._finalize_ranks(reranked_chunks[: request.top_k])
        return RetrievalResult(
            query=request.query,
            chunks=returned_chunks,
            stats=RetrievalStats(
                requested_top_k=request.top_k,
                mode=request.mode,
                reranker=request.reranker,
                raw_result_count=len(chunks),
                dense_result_count=len(dense_chunks),
                bm25_result_count=len(bm25_chunks),
                deduplicated_count=len(deduplicated_chunks),
                returned_count=len(returned_chunks),
                reranked=request.reranker is not RerankerMode.NONE,
            ),
        )

    async def retrieve_async(self, request: RetrievalRequest) -> RetrievalResult:
        """Non-blocking online retrieval; sync local libraries run off-loop."""
        candidate_top_k = self._candidate_top_k(request.top_k)
        dense_chunks, bm25_chunks = await asyncio.gather(
            self._dense_search_async(request, candidate_top_k),
            self._bm25_search_async(request, candidate_top_k),
        )
        chunks = self._combine_results(request, dense_chunks, bm25_chunks)
        deduplicated_chunks = self._deduplicate(chunks)
        reranked_chunks = await asyncio.to_thread(
            self._rerank,
            request.query,
            deduplicated_chunks,
            mode=request.reranker,
        )
        returned_chunks = self._finalize_ranks(reranked_chunks[: request.top_k])
        return RetrievalResult(
            query=request.query,
            chunks=returned_chunks,
            stats=RetrievalStats(
                requested_top_k=request.top_k,
                mode=request.mode,
                reranker=request.reranker,
                raw_result_count=len(chunks),
                dense_result_count=len(dense_chunks),
                bm25_result_count=len(bm25_chunks),
                deduplicated_count=len(deduplicated_chunks),
                returned_count=len(returned_chunks),
                reranked=request.reranker is not RerankerMode.NONE,
            ),
        )

    async def _dense_search_async(
        self,
        request: RetrievalRequest,
        candidate_top_k: int,
    ) -> List[RetrievedKnowledgeChunk]:
        if request.mode is RetrievalMode.BM25:
            return []
        embedding_provider = self._embedding_provider()
        vector_store = self._vector_store()
        embed_async = getattr(embedding_provider, "embed_text_async", None)
        if embed_async is not None:
            query_embedding = embed_async(
                request.query,
                task_type="RETRIEVAL_QUERY",
            )
            if inspect.isawaitable(query_embedding):
                query_embedding = await query_embedding
        else:
            query_embedding = await asyncio.to_thread(
                embedding_provider.embed_text,
                request.query,
                task_type="RETRIEVAL_QUERY",
            )
        query_async = getattr(vector_store, "query_async", None)
        if query_async is not None:
            chunks = query_async(
                query_embedding,
                top_k=candidate_top_k,
                where=self._build_where_filter(request.filters),
            )
            resolved = await chunks if inspect.isawaitable(chunks) else chunks
            return self._annotate_dense(resolved)
        chunks = await asyncio.to_thread(
            vector_store.query,
            query_embedding,
            top_k=candidate_top_k,
            where=self._build_where_filter(request.filters),
        )
        return self._annotate_dense(chunks)

    async def _bm25_search_async(
        self,
        request: RetrievalRequest,
        candidate_top_k: int,
    ) -> List[RetrievedKnowledgeChunk]:
        if request.mode is RetrievalMode.DENSE:
            return []
        return await asyncio.to_thread(
            self._bm25_search,
            request,
            candidate_top_k,
        )

    def _dense_search(
        self,
        request: RetrievalRequest,
        candidate_top_k: int,
    ) -> List[RetrievedKnowledgeChunk]:
        if request.mode is RetrievalMode.BM25:
            return []
        query_embedding = self._embedding_provider().embed_text(
            request.query,
            task_type="RETRIEVAL_QUERY",
        )
        chunks = self._vector_store().query(
            query_embedding,
            top_k=candidate_top_k,
            where=self._build_where_filter(request.filters),
        )
        return self._annotate_dense(chunks)

    def _bm25_search(
        self,
        request: RetrievalRequest,
        candidate_top_k: int,
    ) -> List[RetrievedKnowledgeChunk]:
        if request.mode is RetrievalMode.DENSE:
            return []
        lexical_index = self._lexical_index()
        chunks = lexical_index.search(
            request.query,
            top_k=candidate_top_k,
            filters=request.filters,
        )
        for rank, chunk in enumerate(chunks, start=1):
            chunk.bm25_rank = chunk.bm25_rank or rank
        return chunks

    def _lexical_index(self) -> KnowledgeLexicalIndex:
        if self.lexical_index is None:
            index_path = Path(settings.lexical_index_path)
            if index_path.exists():
                persisted_index = SQLiteFTS5KnowledgeIndex(index_path)
                if DEFAULT_CHUNKS_PATH.exists():
                    expected_digest = SQLiteFTS5KnowledgeIndex.digest_file(DEFAULT_CHUNKS_PATH)
                    if persisted_index.manifest().get("source_digest") != expected_digest:
                        raise ValueError(
                            "Lexical index is stale. Run scripts/build_lexical_index.py."
                        )
                self.lexical_index = persisted_index
            else:
                self.lexical_index = SQLiteFTS5KnowledgeIndex(index_path)
        return self.lexical_index

    def _embedding_provider(self) -> QueryEmbeddingProvider:
        if self.embedding_provider is None:
            self.embedding_provider = GeminiEmbeddingProvider()
        return self.embedding_provider

    def _vector_store(self) -> KnowledgeVectorStore:
        if self.vector_store is None:
            self.vector_store = ChromaVectorStore()
        return self.vector_store

    def _combine_results(
        self,
        request: RetrievalRequest,
        dense_chunks: List[RetrievedKnowledgeChunk],
        bm25_chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        if request.mode is RetrievalMode.DENSE:
            return dense_chunks
        if request.mode is RetrievalMode.BM25:
            return bm25_chunks
        return self._rank_fusion(
            dense_chunks,
            bm25_chunks,
            dense_weight=request.dense_weight,
            bm25_weight=request.bm25_weight,
        )

    def _rank_fusion(
        self,
        dense_chunks: List[RetrievedKnowledgeChunk],
        bm25_chunks: List[RetrievedKnowledgeChunk],
        *,
        dense_weight: float,
        bm25_weight: float,
    ) -> List[RetrievedKnowledgeChunk]:
        chunks_by_id: Dict[str, RetrievedKnowledgeChunk] = {}
        scores: Dict[str, float] = {}
        for rank, chunk in enumerate(dense_chunks, start=1):
            dense = chunk.model_copy(deep=True)
            dense.dense_distance = chunk.dense_distance if chunk.dense_distance is not None else chunk.distance
            dense.dense_rank = rank
            chunks_by_id.setdefault(chunk.chunk_id, dense)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                dense_weight / (RRF_K + rank)
            )
        for rank, chunk in enumerate(bm25_chunks, start=1):
            if chunk.chunk_id not in chunks_by_id:
                chunks_by_id[chunk.chunk_id] = chunk.model_copy(deep=True)
            combined = chunks_by_id[chunk.chunk_id]
            combined.bm25_score = chunk.bm25_score
            combined.bm25_rank = chunk.bm25_rank or rank
            combined.metadata = {**combined.metadata, **chunk.metadata}
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                bm25_weight / (RRF_K + rank)
            )

        ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        fused_chunks = []
        for fusion_rank, chunk_id in enumerate(ranked_ids, start=1):
            chunk = chunks_by_id[chunk_id]
            chunk.fusion_score = scores[chunk_id]
            chunk.fusion_rank = fusion_rank
            chunk.metadata = {
                **chunk.metadata,
                "hybrid_score": f"{scores[chunk_id]:.6f}",
            }
            fused_chunks.append(chunk)
        return fused_chunks

    def _annotate_dense(
        self,
        chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        for rank, chunk in enumerate(chunks, start=1):
            chunk.dense_distance = chunk.distance
            chunk.dense_rank = rank
        return chunks

    def _finalize_ranks(
        self,
        chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        for rank, chunk in enumerate(chunks, start=1):
            chunk.final_rank = rank
        return chunks

    def _candidate_top_k(self, requested_top_k: int) -> int:
        return max(requested_top_k, requested_top_k * self.candidate_multiplier)

    def _build_where_filter(
        self,
        filters: RetrievalFilters,
    ) -> Optional[Dict[str, object]]:
        conditions: List[Dict[str, object]] = []
        if filters.source_ids:
            conditions.append(self._field_condition("source_id", filters.source_ids))
        if filters.source_types:
            conditions.append(
                self._field_condition(
                    "source_type",
                    [source_type.value for source_type in filters.source_types],
                )
            )
        if filters.versions:
            conditions.append(self._field_condition("version", filters.versions))

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _field_condition(self, field: str, values: List[str]) -> Dict[str, object]:
        if len(values) == 1:
            return {field: values[0]}
        return {field: {"$in": values}}

    def _deduplicate(
        self,
        chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        seen: Set[Tuple[str, str, str, int, str]] = set()
        deduplicated: List[RetrievedKnowledgeChunk] = []
        for chunk in chunks:
            key = self._dedupe_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(chunk)
        return deduplicated

    def _dedupe_key(self, chunk: RetrievedKnowledgeChunk) -> Tuple[str, str, str, int, str]:
        citation = chunk.citation
        return (
            chunk.content_hash,
            citation.source_id,
            citation.version,
            citation.page or 0,
            citation.section or "",
        )

    def _rerank(
        self,
        query: str,
        chunks: List[RetrievedKnowledgeChunk],
        *,
        mode: RerankerMode,
    ) -> List[RetrievedKnowledgeChunk]:
        if mode is RerankerMode.NONE:
            return chunks
        return self._cross_encoder_reranker().rerank(query, chunks)

    def _cross_encoder_reranker(self) -> KnowledgeReranker:
        if self.cross_encoder_reranker is None:
            self.cross_encoder_reranker = CrossEncoderReranker()
        return self.cross_encoder_reranker
