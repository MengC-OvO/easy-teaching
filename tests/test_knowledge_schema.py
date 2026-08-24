import pytest
from pydantic import ValidationError

from app.schemas import (
    CitationMetadata,
    IngestionResult,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    RetrievalFilters,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStats,
)


def make_document() -> KnowledgeDocument:
    return KnowledgeDocument(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        title="EYLF V2.0",
        version="2.0",
        uri="https://example.test/eylf-v2.pdf",
    )


def test_knowledge_chunk_builds_stable_ids_and_hashes() -> None:
    document = make_document()

    first = KnowledgeChunk.from_document(
        document=document,
        section="Learning Outcome 1",
        page=12,
        content=" Children have a strong sense of identity. ",
    )
    second = KnowledgeChunk.from_document(
        document=document,
        section="Learning Outcome 1",
        page=12,
        content="Children have a strong sense of identity.",
    )

    assert first.chunk_id == second.chunk_id
    assert first.content_hash == second.content_hash
    assert first.content == "Children have a strong sense of identity."


def test_knowledge_chunk_exposes_citation_metadata() -> None:
    chunk = KnowledgeChunk.from_document(
        document=make_document(),
        section="Learning Outcome 1",
        page=12,
        content="Children have a strong sense of identity.",
    )

    citation = chunk.citation

    assert citation.source_id == "eylf-v2"
    assert citation.source_type is KnowledgeSourceType.OFFICIAL
    assert citation.title == "EYLF V2.0"
    assert citation.version == "2.0"
    assert citation.section == "Learning Outcome 1"
    assert citation.page == 12


def test_knowledge_chunk_requires_section_or_page() -> None:
    with pytest.raises(ValidationError):
        KnowledgeChunk.from_document(
            document=make_document(),
            content="A chunk without a citable location.",
        )


def test_knowledge_document_marks_synthetic_sources() -> None:
    document = KnowledgeDocument(
        source_id="synthetic-centre-policy",
        source_type=KnowledgeSourceType.SYNTHETIC,
        title="Synthetic Centre Outdoor Play Policy",
        version="2026.07",
    )

    assert document.source_type is KnowledgeSourceType.SYNTHETIC


def test_ingestion_result_chunk_count_must_match() -> None:
    chunk = KnowledgeChunk.from_document(
        document=make_document(),
        section="Learning Outcome 1",
        content="Children have a strong sense of identity.",
    )

    with pytest.raises(ValidationError):
        IngestionResult(source_id="eylf-v2", chunk_count=2, chunks=[chunk])

    result = IngestionResult(source_id="eylf-v2", chunk_count=1, chunks=[chunk])

    assert result.chunk_count == 1
    assert result.chunks == [chunk]


def test_retrieval_request_supports_filters() -> None:
    request = RetrievalRequest(
        query="play-based learning",
        top_k=8,
        filters=RetrievalFilters(
            source_ids=["eylf-v2"],
            source_types=[KnowledgeSourceType.OFFICIAL],
            versions=["2.0-2022"],
        ),
    )

    assert request.query == "play-based learning"
    assert request.top_k == 8
    assert request.filters.source_ids == ["eylf-v2"]
    assert request.filters.source_types == [KnowledgeSourceType.OFFICIAL]
    assert request.filters.versions == ["2.0-2022"]


def test_retrieval_request_limits_top_k() -> None:
    with pytest.raises(ValidationError):
        RetrievalRequest(query="policy", top_k=0)

    with pytest.raises(ValidationError):
        RetrievalRequest(query="policy", top_k=21)


def test_retrieval_result_exposes_citations() -> None:
    citation = CitationMetadata(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        title="EYLF V2.0",
        version="2.0-2022",
        page=21,
    )
    chunk = RetrievedKnowledgeChunk(
        chunk_id="chunk-1",
        content="Play-based learning provides opportunities for children.",
        citation=citation,
        content_hash="a" * 64,
        distance=0.2,
    )
    result = RetrievalResult(
        query="play-based learning",
        chunks=[chunk],
        stats=RetrievalStats(
            requested_top_k=5,
            raw_result_count=1,
            deduplicated_count=1,
            returned_count=1,
        ),
    )

    assert result.citations == [citation]
