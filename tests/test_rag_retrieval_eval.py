from app.schemas import (
    CitationMetadata,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeScope,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    RetrievalMode,
    RetrievalResult,
    RetrievalStats,
    RerankerMode,
)
from evals.rag_retrieval import (
    RagGoldEvidence,
    RagRetrievalCase,
    evaluate_case,
    percentile,
    summarize_mode,
)


def make_chunk(source_id: str, page: int, section: str, content: str) -> KnowledgeChunk:
    return KnowledgeChunk.from_document(
        document=KnowledgeDocument(
            source_id=source_id,
            source_type=KnowledgeSourceType.OFFICIAL,
            title=source_id,
            version="1",
            uri=f"https://example.test/{source_id}.pdf",
        ),
        content=content,
        page=page,
        section=section,
    )


def retrieved(chunk: KnowledgeChunk, distance: float) -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        citation=CitationMetadata(**chunk.citation.model_dump()),
        content_hash=chunk.content_hash,
        distance=distance,
    )


def test_evaluate_case_computes_rank_metrics_scope_and_citation_integrity() -> None:
    relevant_first = make_chunk("eylf-v2", 21, "Play-based learning", "First evidence")
    wrong_scope = make_chunk("nqs-guide-qa1", 100, "Other", "Other evidence")
    relevant_second = make_chunk("eylf-v2", 22, "Play-based learning", "Second evidence")
    chunks = [relevant_first, wrong_scope, relevant_second]
    result = RetrievalResult(
        query="play",
        chunks=[retrieved(chunk, index / 10) for index, chunk in enumerate(chunks, start=1)],
        stats=RetrievalStats(
            requested_top_k=3,
            mode=RetrievalMode.HYBRID,
            reranker=RerankerMode.NONE,
            raw_result_count=3,
            deduplicated_count=3,
            returned_count=3,
        ),
    )
    case = RagRetrievalCase(
        case_id="play",
        query="play",
        scope=KnowledgeScope.EYLF,
        relevant_evidence=[
            RagGoldEvidence(
                source_id="eylf-v2", page=21, section_contains="Play-based", relevance=3
            ),
            RagGoldEvidence(
                source_id="eylf-v2", page=22, section_contains="Play-based", relevance=2
            ),
        ],
    )

    measured = evaluate_case(
        case=case,
        mode="hybrid",
        result=result,
        latency_ms=12.0,
        ks=[1, 3],
        chunk_catalog={chunk.chunk_id: chunk for chunk in chunks},
    )

    assert measured.recall_at_k == {1: 0.5, 3: 1.0}
    assert measured.reciprocal_rank == 1.0
    assert measured.ndcg_at_k[1] == 1.0
    assert 0.8 < measured.ndcg_at_k[3] < 1.0
    assert measured.scope_violation_count == 1
    assert measured.citation_correct_count == 3

    summary = summarize_mode([measured], [1, 3])
    assert summary.scope_violation_rate == 1 / 3
    assert summary.citation_correctness == 1.0
    assert summary.latency_p95_ms == 12.0


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0
