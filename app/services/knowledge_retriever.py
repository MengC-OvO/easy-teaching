import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, Tuple

from app.schemas import (
    KnowledgeChunk,
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
from app.services.knowledge_ingestion import KnowledgeIngestionService
from app.services.vector_store import ChromaVectorStore


DEFAULT_CHUNKS_PATH = Path("data/knowledge/processed/chunks.jsonl")
RRF_K = 60


class QueryEmbeddingProvider(Protocol):
    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
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


class BM25KnowledgeIndex:
    def __init__(
        self,
        chunks: List[KnowledgeChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized_chunks = [tokenize(chunk.content) for chunk in chunks]
        self.term_counts = [Counter(tokens) for tokens in self.tokenized_chunks]
        self.document_frequencies = self._document_frequencies(self.tokenized_chunks)
        self.average_document_length = self._average_length(self.tokenized_chunks)

    @classmethod
    def from_jsonl(
        cls,
        chunks_path: Path = DEFAULT_CHUNKS_PATH,
        *,
        project_root: Optional[Path] = None,
    ) -> "BM25KnowledgeIndex":
        ingestion = KnowledgeIngestionService(project_root=project_root or Path.cwd())
        return cls(ingestion.read_chunks_jsonl(chunks_path))

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters,
    ) -> List[RetrievedKnowledgeChunk]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored_chunks: List[Tuple[float, KnowledgeChunk]] = []
        for chunk, term_count, document_length in zip(
            self.chunks,
            self.term_counts,
            [len(tokens) for tokens in self.tokenized_chunks],
        ):
            if not matches_filters(chunk, filters):
                continue
            score = self._score(query_terms, term_count, document_length)
            if score <= 0:
                continue
            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [
            self._to_retrieved_chunk(chunk, score)
            for score, chunk in scored_chunks[:top_k]
        ]

    def _score(
        self,
        query_terms: List[str],
        term_count: Counter,
        document_length: int,
    ) -> float:
        score = 0.0
        total_documents = len(self.chunks)
        for term in query_terms:
            term_frequency = term_count.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self.document_frequencies.get(term, 0)
            idf = math.log(
                1
                + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (
                1
                - self.b
                + self.b * document_length / max(self.average_document_length, 1)
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score

    def _to_retrieved_chunk(
        self,
        chunk: KnowledgeChunk,
        score: float,
    ) -> RetrievedKnowledgeChunk:
        return RetrievedKnowledgeChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            citation=chunk.citation,
            content_hash=chunk.content_hash,
            distance=1 / (1 + score),
            metadata={**chunk.metadata, "bm25_score": f"{score:.6f}"},
        )

    def _document_frequencies(
        self,
        tokenized_chunks: List[List[str]],
    ) -> Dict[str, int]:
        frequencies: Dict[str, int] = {}
        for tokens in tokenized_chunks:
            for token in set(tokens):
                frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies

    def _average_length(self, tokenized_chunks: List[List[str]]) -> float:
        if not tokenized_chunks:
            return 0.0
        return sum(len(tokens) for tokens in tokenized_chunks) / len(tokenized_chunks)


class LexicalReranker:
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        if not chunks:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return chunks

        return sorted(
            chunks,
            key=lambda chunk: (
                lexical_overlap_score(query_terms, chunk.content),
                -chunk.distance,
            ),
            reverse=True,
        )


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
        scored_chunks = list(zip(scores, chunks))
        scored_chunks.sort(key=lambda item: item[0], reverse=True)

        reranked: List[RetrievedKnowledgeChunk] = []
        for score, chunk in scored_chunks:
            chunk.metadata = {
                **chunk.metadata,
                "cross_encoder_score": f"{score:.6f}",
                "cross_encoder_model": self.model_name,
            }
            reranked.append(chunk)
        return reranked


class KnowledgeRetriever:
    def __init__(
        self,
        *,
        embedding_provider: Optional[QueryEmbeddingProvider] = None,
        vector_store: Optional[KnowledgeVectorStore] = None,
        lexical_index: Optional[KnowledgeLexicalIndex] = None,
        lexical_reranker: Optional[KnowledgeReranker] = None,
        cross_encoder_reranker: Optional[KnowledgeReranker] = None,
        candidate_multiplier: int = 3,
    ) -> None:
        self.embedding_provider = embedding_provider or GeminiEmbeddingProvider()
        self.vector_store = vector_store or ChromaVectorStore()
        self.lexical_index = lexical_index
        self.lexical_reranker = lexical_reranker or LexicalReranker()
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
        returned_chunks = reranked_chunks[: request.top_k]
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
                reranked=request.use_reranker,
            ),
        )

    def _dense_search(
        self,
        request: RetrievalRequest,
        candidate_top_k: int,
    ) -> List[RetrievedKnowledgeChunk]:
        if request.mode is RetrievalMode.BM25:
            return []
        query_embedding = self.embedding_provider.embed_text(
            request.query,
            task_type="RETRIEVAL_QUERY",
        )
        return self.vector_store.query(
            query_embedding,
            top_k=candidate_top_k,
            where=self._build_where_filter(request.filters),
        )

    def _bm25_search(
        self,
        request: RetrievalRequest,
        candidate_top_k: int,
    ) -> List[RetrievedKnowledgeChunk]:
        if request.mode is RetrievalMode.DENSE:
            return []
        lexical_index = self._lexical_index()
        return lexical_index.search(
            request.query,
            top_k=candidate_top_k,
            filters=request.filters,
        )

    def _lexical_index(self) -> KnowledgeLexicalIndex:
        if self.lexical_index is None:
            self.lexical_index = BM25KnowledgeIndex.from_jsonl()
        return self.lexical_index

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
            chunks_by_id.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                dense_weight / (RRF_K + rank)
            )
        for rank, chunk in enumerate(bm25_chunks, start=1):
            chunks_by_id.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (
                bm25_weight / (RRF_K + rank)
            )

        ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
        fused_chunks = []
        for chunk_id in ranked_ids:
            chunk = chunks_by_id[chunk_id]
            chunk.metadata = {
                **chunk.metadata,
                "hybrid_score": f"{scores[chunk_id]:.6f}",
            }
            fused_chunks.append(chunk)
        return fused_chunks

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
        if mode is RerankerMode.LEXICAL:
            return self.lexical_reranker.rerank(query, chunks)
        return self._cross_encoder_reranker().rerank(query, chunks)

    def _cross_encoder_reranker(self) -> KnowledgeReranker:
        if self.cross_encoder_reranker is None:
            self.cross_encoder_reranker = CrossEncoderReranker()
        return self.cross_encoder_reranker


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_overlap_score(query_terms: Set[str], content: str) -> float:
    content_terms = set(tokenize(content))
    if not query_terms:
        return 0.0
    return len(query_terms & content_terms) / len(query_terms)


def matches_filters(chunk: KnowledgeChunk, filters: RetrievalFilters) -> bool:
    if filters.source_ids and chunk.document.source_id not in filters.source_ids:
        return False
    if filters.source_types and chunk.document.source_type not in filters.source_types:
        return False
    if filters.versions and chunk.document.version not in filters.versions:
        return False
    return True
