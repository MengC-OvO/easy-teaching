from pathlib import Path

from app.schemas import KnowledgeSourceType
from app.services import KnowledgeIngestionService, KnowledgeSourceSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ingestion_service_loads_source_manifest() -> None:
    service = KnowledgeIngestionService(project_root=PROJECT_ROOT)

    sources = service.load_sources(Path("data/knowledge/sources.json"))

    assert [source.source_id for source in sources] == [
        "eylf-v2",
        "nqs-guide-qa1",
        "synthetic-centre-policies",
    ]
    assert sources[0].source_type is KnowledgeSourceType.OFFICIAL
    assert sources[-1].source_type is KnowledgeSourceType.SYNTHETIC


def test_ingestion_service_chunks_markdown_by_section(tmp_path) -> None:
    markdown = tmp_path / "policy.md"
    markdown.write_text(
        "\n".join(
            [
                "# Synthetic Policy",
                "",
                "Intro content.",
                "",
                "## Outdoor Play",
                "",
                "Children explore outdoors with educator supervision.",
                "",
                "## Family Drafts",
                "",
                "Messages remain drafts until teacher approval.",
            ]
        ),
        encoding="utf-8",
    )
    source = KnowledgeSourceSpec(
        source_id="synthetic-test",
        source_type=KnowledgeSourceType.SYNTHETIC,
        title="Synthetic Policy",
        version="2026.07",
        path=str(markdown),
        format="markdown",
    )
    service = KnowledgeIngestionService(project_root=PROJECT_ROOT)

    result = service.ingest_source(source)

    assert result.source_id == "synthetic-test"
    assert result.chunk_count == 3
    assert [chunk.section for chunk in result.chunks] == [
        "Synthetic Policy",
        "Outdoor Play",
        "Family Drafts",
    ]
    assert all(chunk.document.source_type is KnowledgeSourceType.SYNTHETIC for chunk in result.chunks)


def test_ingestion_service_extracts_pdf_pages_with_page_metadata() -> None:
    source = KnowledgeSourceSpec(
        source_id="eylf-test",
        source_type=KnowledgeSourceType.OFFICIAL,
        title="EYLF V2.0",
        version="2.0-2022",
        uri="https://example.test/eylf.pdf",
        path="data/knowledge/raw/official/eylf-v2.0.pdf",
        format="pdf",
    )
    service = KnowledgeIngestionService(project_root=PROJECT_ROOT, chunk_size=2000)

    result = service.ingest_source(source, max_pages_per_pdf=2)

    assert result.source_id == "eylf-test"
    assert result.chunk_count >= 1
    assert all(chunk.page in {1, 2} for chunk in result.chunks)
    assert all(chunk.citation.title == "EYLF V2.0" for chunk in result.chunks)
    assert all(chunk.citation.version == "2.0-2022" for chunk in result.chunks)


def test_ingestion_service_chunk_ids_are_stable(tmp_path) -> None:
    markdown = tmp_path / "policy.md"
    markdown.write_text("# Policy\n\nStable content.", encoding="utf-8")
    source = KnowledgeSourceSpec(
        source_id="synthetic-test",
        source_type=KnowledgeSourceType.SYNTHETIC,
        title="Synthetic Policy",
        version="2026.07",
        path=str(markdown),
        format="markdown",
    )
    service = KnowledgeIngestionService(project_root=PROJECT_ROOT)

    first = service.ingest_source(source)
    second = service.ingest_source(source)

    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    assert [chunk.content_hash for chunk in first.chunks] == [
        chunk.content_hash for chunk in second.chunks
    ]


def test_ingestion_service_writes_and_reads_chunks_jsonl(tmp_path) -> None:
    markdown = tmp_path / "policy.md"
    markdown.write_text(
        "# Policy\n\nStable content.\n\n## Section Two\n\nMore stable content.",
        encoding="utf-8",
    )
    source = KnowledgeSourceSpec(
        source_id="synthetic-test",
        source_type=KnowledgeSourceType.SYNTHETIC,
        title="Synthetic Policy",
        version="2026.07",
        path=str(markdown),
        format="markdown",
    )
    output = tmp_path / "processed" / "chunks.jsonl"
    service = KnowledgeIngestionService(project_root=PROJECT_ROOT)
    result = service.ingest_source(source)

    written = service.write_chunks_jsonl([result], output)
    loaded = service.read_chunks_jsonl(output)

    assert written == result.chunk_count
    assert len(output.read_text(encoding="utf-8").splitlines()) == result.chunk_count
    assert loaded == result.chunks


def test_ingestion_service_ignores_blank_lines_when_reading_jsonl(tmp_path) -> None:
    markdown = tmp_path / "policy.md"
    markdown.write_text("# Policy\n\nStable content.", encoding="utf-8")
    source = KnowledgeSourceSpec(
        source_id="synthetic-test",
        source_type=KnowledgeSourceType.SYNTHETIC,
        title="Synthetic Policy",
        version="2026.07",
        path=str(markdown),
        format="markdown",
    )
    service = KnowledgeIngestionService(project_root=PROJECT_ROOT)
    result = service.ingest_source(source)
    output = tmp_path / "chunks.jsonl"
    service.write_chunks_jsonl([result], output)
    output.write_text(output.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    loaded = service.read_chunks_jsonl(output)

    assert loaded == result.chunks


def test_ingestion_service_rejects_unknown_source_format(tmp_path) -> None:
    file_path = tmp_path / "policy.txt"
    file_path.write_text("Plain text.", encoding="utf-8")
    source = KnowledgeSourceSpec(
        source_id="unknown-format",
        source_type=KnowledgeSourceType.SYNTHETIC,
        title="Unknown",
        version="1",
        path=str(file_path),
        format="txt",
    )
    service = KnowledgeIngestionService(project_root=PROJECT_ROOT)

    try:
        service.ingest_source(source)
    except ValueError as error:
        assert "Unsupported knowledge source format" in str(error)
    else:
        raise AssertionError("Unknown source format should fail")
