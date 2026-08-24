from app.schemas import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeSourceType,
    RetrievalFilters,
)
from app.services import SQLiteFTS5KnowledgeIndex


def make_chunk(content: str, *, source_id: str = "eylf-v2") -> KnowledgeChunk:
    return KnowledgeChunk.from_document(
        document=KnowledgeDocument(
            source_id=source_id,
            source_type=KnowledgeSourceType.OFFICIAL,
            title="EYLF V2.0",
            version="2.0-2022",
        ),
        content=content,
        section="Play-based learning",
        page=21,
    )


def test_sqlite_fts5_index_persists_and_returns_bm25_scores(tmp_path) -> None:
    path = tmp_path / "knowledge.sqlite3"
    play = make_chunk("Children learn through outdoor sensory play.")
    family = make_chunk("Families receive reviewed communication drafts.", source_id="policy")

    built = SQLiteFTS5KnowledgeIndex.build(path, [play, family], source_digest="digest")
    reopened = SQLiteFTS5KnowledgeIndex(path)
    results = reopened.search("sensory play", top_k=5, filters=RetrievalFilters())

    assert built.count() == 2
    assert reopened.manifest()["source_digest"] == "digest"
    assert results[0].chunk_id == play.chunk_id
    assert results[0].bm25_score is not None
    assert results[0].bm25_rank == 1


def test_sqlite_fts5_index_applies_metadata_filters(tmp_path) -> None:
    path = tmp_path / "knowledge.sqlite3"
    eylf = make_chunk("Play supports learning.")
    policy = make_chunk("Play requires supervision.", source_id="policy")
    index = SQLiteFTS5KnowledgeIndex.build(path, [eylf, policy])

    results = index.search(
        "play",
        top_k=5,
        filters=RetrievalFilters(source_ids=["policy"]),
    )

    assert [result.chunk_id for result in results] == [policy.chunk_id]
