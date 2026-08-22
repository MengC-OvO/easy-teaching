from app.config import Settings
from app.schemas import KnowledgeChunk, KnowledgeDocument, KnowledgeSourceType
from app.services import ChromaVectorStore, VectorIndexConfigurationError


def make_settings(tmp_path) -> Settings:
    return Settings(
        EMBEDDING_MODEL_NAME="test-embedding-model",
        EMBEDDING_DIMENSION=3,
        CHROMA_PATH=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME="test_knowledge",
    )


def make_document() -> KnowledgeDocument:
    return KnowledgeDocument(
        source_id="synthetic-centre-policies",
        source_type=KnowledgeSourceType.SYNTHETIC,
        title="Synthetic Centre Policies",
        version="2026.07",
    )


def make_chunk(content: str, section: str) -> KnowledgeChunk:
    return KnowledgeChunk.from_document(
        document=make_document(),
        content=content,
        section=section,
        metadata={"topic": section.lower().replace(" ", "-")},
    )


def test_chroma_vector_store_upserts_and_queries_chunks(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    outdoor = make_chunk(
        "Outdoor play should include educator supervision.",
        "Outdoor Play",
    )
    family = make_chunk(
        "Family communication must remain a reviewed draft.",
        "Family Drafts",
    )

    written = store.upsert_chunks(
        [outdoor, family],
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )
    results = store.query([1.0, 0.0, 0.0], top_k=1)

    assert written == 2
    assert store.count() == 2
    assert len(results) == 1
    assert results[0].chunk_id == outdoor.chunk_id
    assert results[0].content == outdoor.content
    assert results[0].citation.source_id == "synthetic-centre-policies"
    assert results[0].citation.source_type is KnowledgeSourceType.SYNTHETIC
    assert results[0].citation.section == "Outdoor Play"
    assert results[0].metadata == {"topic": "outdoor-play"}


def test_chroma_vector_store_upsert_is_stable_for_existing_ids(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    chunk = make_chunk("Stable content.", "Policy")

    first = store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])
    second = store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])

    assert first == 1
    assert second == 1
    assert store.count() == 1


def test_chroma_vector_store_can_filter_by_metadata(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    outdoor = make_chunk("Outdoor play content.", "Outdoor Play")
    family = make_chunk("Family message content.", "Family Drafts")
    store.upsert_chunks(
        [outdoor, family],
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
    )

    results = store.query(
        [1.0, 0.0, 0.0],
        top_k=2,
        where={"section": "Family Drafts"},
    )

    assert [result.chunk_id for result in results] == [family.chunk_id]


def test_chroma_vector_store_can_filter_by_combined_metadata(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    synthetic = make_chunk("Synthetic policy content.", "Policy")
    official_document = KnowledgeDocument(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        title="EYLF V2.0",
        version="2.0-2022",
    )
    official = KnowledgeChunk.from_document(
        document=official_document,
        content="Official play-based learning content.",
        page=21,
    )
    store.upsert_chunks(
        [synthetic, official],
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
    )

    results = store.query(
        [1.0, 0.0, 0.0],
        top_k=2,
        where={
            "$and": [
                {"source_type": "official"},
                {"version": "2.0-2022"},
            ]
        },
    )

    assert [result.chunk_id for result in results] == [official.chunk_id]


def test_chroma_vector_store_can_report_existing_chunk_ids(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    chunk = make_chunk("Stable content.", "Policy")
    missing = make_chunk("Missing content.", "Policy")
    store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])

    existing_ids = store.existing_chunk_ids([chunk.chunk_id, missing.chunk_id])

    assert existing_ids == {chunk.chunk_id}


def test_chroma_vector_store_rejects_wrong_embedding_dimension(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    chunk = make_chunk("Stable content.", "Policy")

    try:
        store.upsert_chunks([chunk], [[1.0, 0.0]])
    except ValueError as error:
        assert "Embedding dimension does not match settings" in str(error)
    else:
        raise AssertionError("Wrong embedding dimension should fail")


def test_chroma_vector_store_rejects_mismatched_chunk_and_embedding_counts(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    chunk = make_chunk("Stable content.", "Policy")

    try:
        store.upsert_chunks([chunk], [])
    except ValueError as error:
        assert "chunks and embeddings must have the same length" in str(error)
    else:
        raise AssertionError("Mismatched chunk and embedding counts should fail")


def test_chroma_vector_store_exposes_index_metadata(tmp_path) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))

    metadata = store.index_metadata()

    assert metadata.collection_name == "test_knowledge"
    assert metadata.index_method == "hnsw"
    assert metadata.distance_metric == "cosine"
    assert metadata.embedding_model_name == "test-embedding-model"
    assert metadata.embedding_dimension == 3
    assert metadata.index_version == "easyteaching-knowledge-v1"


def test_chroma_vector_store_rejects_existing_collection_with_different_model(
    tmp_path,
) -> None:
    store = ChromaVectorStore(make_settings(tmp_path))
    chunk = make_chunk("Stable content.", "Policy")
    store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])

    changed_settings = Settings(
        EMBEDDING_MODEL_NAME="other-embedding-model",
        EMBEDDING_DIMENSION=3,
        CHROMA_PATH=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME="test_knowledge",
    )

    try:
        ChromaVectorStore(changed_settings)
    except VectorIndexConfigurationError as error:
        assert "embedding_model_name" in str(error)
    else:
        raise AssertionError("Existing collection with a different model should fail")


def test_chroma_vector_store_accepts_existing_collection_with_same_metadata(
    tmp_path,
) -> None:
    settings = make_settings(tmp_path)
    store = ChromaVectorStore(settings)
    chunk = make_chunk("Stable content.", "Policy")
    store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])

    reopened = ChromaVectorStore(settings)

    assert reopened.count() == 1


def test_chroma_vector_store_rejects_existing_collection_with_wrong_distance_metric(
    tmp_path,
) -> None:
    settings = make_settings(tmp_path)
    client = ChromaVectorStore(settings).client
    client.delete_collection(settings.chroma_collection_name)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        configuration={"hnsw": {"space": "l2"}},
        metadata={
            "collection_name": settings.chroma_collection_name,
            "index_method": "hnsw",
            "distance_metric": "l2",
            "embedding_model_name": settings.embedding_model_name,
            "embedding_dimension": settings.embedding_dimension,
            "index_version": "easyteaching-knowledge-v1",
        },
    )
    chunk = make_chunk("Stable content.", "Policy")
    collection.upsert(
        ids=[chunk.chunk_id],
        documents=[chunk.content],
        embeddings=[[1.0, 0.0, 0.0]],
        metadatas=[{"content_hash": chunk.content_hash}],
    )

    try:
        ChromaVectorStore(settings)
    except VectorIndexConfigurationError as error:
        assert "distance_metric" in str(error) or "distance metric" in str(error)
    else:
        raise AssertionError("Existing collection with l2 distance should fail")


def test_chroma_vector_store_can_reset_existing_collection(tmp_path) -> None:
    settings = make_settings(tmp_path)
    store = ChromaVectorStore(settings)
    chunk = make_chunk("Stable content.", "Policy")
    store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])

    reset_store = ChromaVectorStore(settings, reset_collection=True)

    assert reset_store.count() == 0
    assert reset_store.collection.configuration["hnsw"]["space"] == "cosine"
