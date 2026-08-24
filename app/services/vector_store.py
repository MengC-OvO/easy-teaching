from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.errors import NotFoundError

from app.config import Settings, settings
from app.schemas import (
    CitationMetadata,
    KnowledgeChunk,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    VectorIndexMetadata,
)


INDEX_VERSION = "easyteaching-knowledge-v2"
INDEX_METHOD = "hnsw"
DISTANCE_METRIC = "cosine"


class VectorIndexConfigurationError(ValueError):
    """Raised when a Chroma collection was built with incompatible settings."""


class ChromaVectorStore:
    def __init__(
        self,
        provider_settings: Settings = settings,
        client: Optional[Any] = None,
        reset_collection: bool = False,
    ) -> None:
        self.settings = provider_settings
        self.client = client or chromadb.PersistentClient(path=self.settings.chroma_path)
        if reset_collection:
            self._delete_collection_if_exists()
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            configuration=self._collection_configuration(),
            metadata=self._collection_metadata(),
        )
        self._ensure_collection_metadata()
        self._ensure_collection_configuration()

    def upsert_chunks(
        self,
        chunks: List[KnowledgeChunk],
        embeddings: List[List[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0
        self._validate_embeddings(embeddings)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[self._chunk_to_metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def query(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedKnowledgeChunk]:
        self._validate_embeddings([query_embedding])
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return self._query_result_to_chunks(result)

    def count(self) -> int:
        return self.collection.count()

    def existing_chunk_ids(self, chunk_ids: List[str]) -> Set[str]:
        if not chunk_ids:
            return set()
        unique_chunk_ids = list(dict.fromkeys(chunk_ids))
        result = self.collection.get(ids=unique_chunk_ids, include=[])
        return set(result.get("ids", []))

    def prune_to_chunk_ids(self, expected_chunk_ids: Set[str]) -> int:
        """Remove stale vectors after a full source rebuild."""
        stored_ids = set(self.collection.get(include=[]).get("ids", []))
        stale_ids = sorted(stored_ids - expected_chunk_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return len(stale_ids)

    def index_metadata(self) -> VectorIndexMetadata:
        return VectorIndexMetadata(
            collection_name=self.settings.chroma_collection_name,
            index_method=INDEX_METHOD,
            distance_metric=DISTANCE_METRIC,
            embedding_model_name=self.settings.embedding_model_name,
            embedding_dimension=self.settings.embedding_dimension,
            index_version=INDEX_VERSION,
        )

    def _delete_collection_if_exists(self) -> None:
        try:
            self.client.delete_collection(self.settings.chroma_collection_name)
        except NotFoundError:
            return

    def _collection_configuration(self) -> Dict[str, Any]:
        return {
            INDEX_METHOD: {
                "space": DISTANCE_METRIC,
            }
        }

    def _collection_metadata(self) -> Dict[str, Any]:
        metadata = self.index_metadata()
        return metadata.model_dump()

    def _ensure_collection_metadata(self) -> None:
        expected = self._collection_metadata()
        actual = self.collection.metadata or {}
        mismatches = {
            key: (actual.get(key), expected_value)
            for key, expected_value in expected.items()
            if actual.get(key) != expected_value
        }

        if not mismatches:
            return

        if self.collection.count() == 0:
            self.collection.modify(metadata=expected)
            return

        details = ", ".join(
            f"{key}: stored={stored!r}, configured={configured!r}"
            for key, (stored, configured) in mismatches.items()
        )
        raise VectorIndexConfigurationError(
            "Chroma collection metadata does not match the configured vector index. "
            f"Rebuild the collection or use matching settings. {details}"
        )

    def _ensure_collection_configuration(self) -> None:
        configuration = getattr(self.collection, "configuration", None) or {}
        index_configuration = configuration.get(INDEX_METHOD) or {}
        actual_distance_metric = index_configuration.get("space")
        if actual_distance_metric == DISTANCE_METRIC:
            return

        raise VectorIndexConfigurationError(
            "Chroma collection distance metric does not match the configured vector index. "
            f"Rebuild the collection or use matching settings. "
            f"stored={actual_distance_metric!r}, configured={DISTANCE_METRIC!r}"
        )

    def _validate_embeddings(self, embeddings: List[List[float]]) -> None:
        for embedding in embeddings:
            if len(embedding) != self.settings.embedding_dimension:
                raise ValueError(
                    "Embedding dimension does not match settings: "
                    f"expected {self.settings.embedding_dimension}, got {len(embedding)}"
                )

    def _chunk_to_metadata(self, chunk: KnowledgeChunk) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "source_id": chunk.document.source_id,
            "source_type": chunk.document.source_type.value,
            "title": chunk.document.title,
            "version": chunk.document.version,
            "section": chunk.section or "",
            "page": chunk.page or 0,
            "uri": chunk.document.uri or "",
            "content_hash": chunk.content_hash,
        }
        for key, value in chunk.metadata.items():
            metadata[f"custom_{key}"] = value
        return metadata

    def _query_result_to_chunks(self, result: Dict[str, Any]) -> List[RetrievedKnowledgeChunk]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: List[RetrievedKnowledgeChunk] = []
        for chunk_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            chunks.append(
                RetrievedKnowledgeChunk(
                    chunk_id=chunk_id,
                    content=content,
                    citation=self._metadata_to_citation(metadata),
                    content_hash=metadata["content_hash"],
                    distance=distance,
                    dense_distance=distance,
                    metadata=self._custom_metadata(metadata),
                )
            )
        return chunks

    def _metadata_to_citation(self, metadata: Dict[str, Any]) -> CitationMetadata:
        page = metadata.get("page")
        return CitationMetadata(
            source_id=metadata["source_id"],
            source_type=KnowledgeSourceType(metadata["source_type"]),
            title=metadata["title"],
            version=metadata["version"],
            section=metadata.get("section") or None,
            page=page if page else None,
            uri=metadata.get("uri") or None,
        )

    def _custom_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        prefix = "custom_"
        return {
            key[len(prefix) :]: str(value)
            for key, value in metadata.items()
            if key.startswith(prefix)
        }
