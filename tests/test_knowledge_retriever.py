import asyncio
from typing import Dict, List, Optional

from app.schemas import (
    CitationMetadata,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    RerankerMode,
    RetrievalFilters,
    RetrievalMode,
    RetrievalRequest,
)
from app.services import BM25KnowledgeIndex, KnowledgeRetriever


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.calls = []

    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        self.calls.append({"text": text, "task_type": task_type})
        return [0.1, 0.2, 0.3]

    async def embed_text_async(
        self, text: str, *, task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        return self.embed_text(text, task_type=task_type)


class FakeVectorStore:
    def __init__(self, chunks: List[RetrievedKnowledgeChunk]) -> None:
        self.chunks = chunks
        self.calls = []

    def query(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, object]] = None,
    ) -> List[RetrievedKnowledgeChunk]:
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "where": where,
            }
        )
        return self.chunks[:top_k]

    async def query_async(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, object]] = None,
    ) -> List[RetrievedKnowledgeChunk]:
        return self.query(query_embedding, top_k=top_k, where=where)


class FakeCrossEncoderReranker:
    def __init__(self) -> None:
        self.calls = []

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedKnowledgeChunk],
    ) -> List[RetrievedKnowledgeChunk]:
        self.calls.append({"query": query, "chunk_ids": [chunk.chunk_id for chunk in chunks]})
        reranked = list(reversed(chunks))
        for index, chunk in enumerate(reranked, start=1):
            chunk.metadata = {
                **chunk.metadata,
                "cross_encoder_score": f"{1 / index:.6f}",
                "cross_encoder_model": "fake-cross-encoder",
            }
        return reranked


def make_retrieved_chunk(
    chunk_id: str,
    distance: float,
    *,
    content_hash: str = "a" * 64,
) -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        chunk_id=chunk_id,
        content="Play-based learning provides opportunities for children.",
        citation=CitationMetadata(
            source_id="eylf-v2",
            source_type=KnowledgeSourceType.OFFICIAL,
            title="EYLF V2.0",
            version="2.0-2022",
            page=21,
        ),
        content_hash=content_hash,
        distance=distance,
    )


def make_knowledge_chunk(
    content: str,
    *,
    source_id: str = "eylf-v2",
    version: str = "2.0-2022",
) -> KnowledgeChunk:
    return KnowledgeChunk.from_document(
        document=KnowledgeDocument(
            source_id=source_id,
            source_type=KnowledgeSourceType.OFFICIAL,
            title="EYLF V2.0",
            version=version,
        ),
        content=content,
        page=21,
    )


def test_knowledge_retriever_runs_dense_top_k_query() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore(
        [
            make_retrieved_chunk("chunk-1", 0.1),
            make_retrieved_chunk("chunk-2", 0.2),
        ]
    )
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        candidate_multiplier=1,
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="What does EYLF say about play?",
            top_k=1,
        )
    )

    assert embedding_provider.calls == [
        {
            "text": "What does EYLF say about play?",
            "task_type": "RETRIEVAL_QUERY",
        }
    ]
    assert vector_store.calls == [
        {
            "query_embedding": [0.1, 0.2, 0.3],
            "top_k": 1,
            "where": None,
        }
    ]
    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-1"]
    assert result.stats.requested_top_k == 1
    assert result.stats.raw_result_count == 1
    assert result.stats.deduplicated_count == 1
    assert result.stats.returned_count == 1
    assert result.citations[0].source_id == "eylf-v2"


def test_knowledge_retriever_async_path_uses_async_dependencies() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore([make_retrieved_chunk("chunk-1", 0.1)])
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        candidate_multiplier=1,
    )

    result = asyncio.run(
        retriever.retrieve_async(RetrievalRequest(query="play", top_k=1))
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-1"]
    assert embedding_provider.calls[0]["task_type"] == "RETRIEVAL_QUERY"
    assert vector_store.calls[0]["top_k"] == 1


def test_knowledge_retriever_builds_chroma_where_filter() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore([make_retrieved_chunk("chunk-1", 0.1)])
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        candidate_multiplier=1,
    )

    retriever.retrieve(
        RetrievalRequest(
            query="play",
            top_k=1,
            filters=RetrievalFilters(
                source_ids=["eylf-v2"],
                source_types=[KnowledgeSourceType.OFFICIAL],
                versions=["2.0-2022"],
            ),
        )
    )

    assert vector_store.calls[0]["where"] == {
        "$and": [
            {"source_id": "eylf-v2"},
            {"source_type": "official"},
            {"version": "2.0-2022"},
        ]
    }


def test_knowledge_retriever_deduplicates_chunks_before_returning_top_k() -> None:
    embedding_provider = FakeEmbeddingProvider()
    duplicate = make_retrieved_chunk("chunk-duplicate", 0.2)
    duplicate.content_hash = "a" * 64
    vector_store = FakeVectorStore(
        [
            make_retrieved_chunk("chunk-1", 0.1),
            duplicate,
            make_retrieved_chunk("chunk-2", 0.3, content_hash="b" * 64),
        ]
    )
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        candidate_multiplier=3,
    )

    result = retriever.retrieve(RetrievalRequest(query="play", top_k=2))

    assert vector_store.calls[0]["top_k"] == 6
    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-1", "chunk-2"]
    assert result.stats.requested_top_k == 2
    assert result.stats.raw_result_count == 3
    assert result.stats.deduplicated_count == 2
    assert result.stats.returned_count == 2


def test_bm25_mode_searches_lexical_index_without_dense_embedding() -> None:
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore([])
    lexical_index = BM25KnowledgeIndex(
        [
            make_knowledge_chunk("Play-based learning supports children learning through play."),
            make_knowledge_chunk("Family communication drafts should be reviewed."),
        ]
    )
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        lexical_index=lexical_index,
        candidate_multiplier=1,
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="play learning",
            top_k=1,
            mode=RetrievalMode.BM25,
        )
    )

    assert embedding_provider.calls == []
    assert vector_store.calls == []
    assert "play" in result.chunks[0].content.lower()
    assert result.chunks[0].metadata["bm25_score"]
    assert result.stats.mode is RetrievalMode.BM25
    assert result.stats.bm25_result_count == 1
    assert result.stats.dense_result_count == 0


def test_hybrid_mode_combines_dense_and_bm25_results() -> None:
    embedding_provider = FakeEmbeddingProvider()
    dense_chunk = make_retrieved_chunk("dense-chunk", 0.1, content_hash="d" * 64)
    vector_store = FakeVectorStore([dense_chunk])
    lexical_index = BM25KnowledgeIndex(
        [
            make_knowledge_chunk("Outdoor sensory mud play supports investigation."),
        ]
    )
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        lexical_index=lexical_index,
        candidate_multiplier=1,
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="outdoor sensory play",
            top_k=2,
            mode=RetrievalMode.HYBRID,
        )
    )

    assert embedding_provider.calls
    assert vector_store.calls
    assert result.stats.mode is RetrievalMode.HYBRID
    assert result.stats.dense_result_count == 1
    assert result.stats.bm25_result_count == 1
    assert result.stats.returned_count == 2
    assert all("hybrid_score" in chunk.metadata for chunk in result.chunks)


def test_reranker_reorders_candidates_by_lexical_overlap() -> None:
    embedding_provider = FakeEmbeddingProvider()
    weak = make_retrieved_chunk("weak", 0.1, content_hash="w" * 64)
    weak.content = "General early childhood guidance."
    strong = make_retrieved_chunk("strong", 0.4, content_hash="s" * 64)
    strong.content = "Outdoor sensory play supports investigation and learning."
    vector_store = FakeVectorStore([weak, strong])
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        candidate_multiplier=1,
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="outdoor sensory play",
            top_k=2,
            use_reranker=True,
        )
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["strong", "weak"]
    assert result.stats.reranked is True
    assert result.stats.reranker is RerankerMode.LEXICAL


def test_cross_encoder_reranker_is_used_when_requested() -> None:
    embedding_provider = FakeEmbeddingProvider()
    first = make_retrieved_chunk("first", 0.1, content_hash="f" * 64)
    second = make_retrieved_chunk("second", 0.2, content_hash="e" * 64)
    vector_store = FakeVectorStore([first, second])
    cross_encoder_reranker = FakeCrossEncoderReranker()
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        cross_encoder_reranker=cross_encoder_reranker,
        candidate_multiplier=1,
    )

    result = retriever.retrieve(
        RetrievalRequest(
            query="play-based learning",
            top_k=2,
            reranker=RerankerMode.CROSS_ENCODER,
        )
    )

    assert cross_encoder_reranker.calls == [
        {"query": "play-based learning", "chunk_ids": ["first", "second"]}
    ]
    assert [chunk.chunk_id for chunk in result.chunks] == ["second", "first"]
    assert result.chunks[0].metadata["cross_encoder_model"] == "fake-cross-encoder"
    assert result.stats.reranked is True
    assert result.stats.reranker is RerankerMode.CROSS_ENCODER
