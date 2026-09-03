import hashlib
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class KnowledgeSourceType(str, Enum):
    OFFICIAL = "official"
    SYNTHETIC = "synthetic"
    CENTRE = "centre"


class KnowledgeScope(str, Enum):
    ALL = "all"
    EYLF = "eylf"
    NQS = "nqs"
    CENTRE_POLICY = "centre_policy"


KNOWLEDGE_SCOPE_SOURCE_IDS = {
    KnowledgeScope.ALL: (),
    KnowledgeScope.EYLF: ("eylf-v2",),
    KnowledgeScope.NQS: ("nqs-guide-qa1",),
    KnowledgeScope.CENTRE_POLICY: ("synthetic-centre-policies",),
}


def source_ids_for_scope(scope: KnowledgeScope) -> List[str]:
    """Resolve a teacher-facing knowledge scope to indexed source IDs."""
    return list(KNOWLEDGE_SCOPE_SOURCE_IDS[scope])


class RetrievalMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RerankerMode(str, Enum):
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"


class KnowledgeDocument(BaseModel):
    source_id: str = Field(min_length=1)
    source_type: KnowledgeSourceType
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    uri: Optional[str] = None


class CitationMetadata(BaseModel):
    source_id: str = Field(min_length=1)
    source_type: KnowledgeSourceType
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    section: Optional[str] = None
    page: Optional[int] = Field(default=None, ge=1)
    uri: Optional[str] = None


class KnowledgeChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    document: KnowledgeDocument
    content: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
    section: Optional[str] = None
    page: Optional[int] = Field(default=None, ge=1)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_document(
        cls,
        *,
        document: KnowledgeDocument,
        content: str,
        section: Optional[str] = None,
        page: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "KnowledgeChunk":
        normalized_content = content.strip()
        content_hash = stable_hash(normalized_content)
        chunk_id = stable_hash(
            "|".join(
                [
                    document.source_id,
                    document.version,
                    section or "",
                    str(page or ""),
                    content_hash,
                ]
            )
        )
        return cls(
            chunk_id=chunk_id,
            document=document,
            content=normalized_content,
            content_hash=content_hash,
            section=section,
            page=page,
            metadata=metadata or {},
        )

    @property
    def citation(self) -> CitationMetadata:
        return CitationMetadata(
            source_id=self.document.source_id,
            source_type=self.document.source_type,
            title=self.document.title,
            version=self.document.version,
            section=self.section,
            page=self.page,
            uri=self.document.uri,
        )

    @property
    def retrieval_text(self) -> str:
        """Text indexed for retrieval; the original content remains citation-ready."""
        context = [self.document.title]
        if self.section:
            context.append(self.section)
        context.append(self.content)
        return "\n".join(context)

    @model_validator(mode="after")
    def validate_citable_location(self) -> "KnowledgeChunk":
        if self.section is None and self.page is None:
            raise ValueError("KnowledgeChunk must include a section or page")
        return self


class IngestionResult(BaseModel):
    source_id: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)
    chunks: List[KnowledgeChunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_chunk_count(self) -> "IngestionResult":
        if self.chunk_count != len(self.chunks):
            raise ValueError("chunk_count must match the number of chunks")
        return self


class VectorIndexMetadata(BaseModel):
    collection_name: str = Field(min_length=1)
    index_method: str = Field(min_length=1)
    distance_metric: str = Field(min_length=1)
    embedding_model_name: str = Field(min_length=1)
    embedding_dimension: int = Field(ge=1)
    index_version: str = Field(min_length=1)


class RetrievedKnowledgeChunk(BaseModel):
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    citation: CitationMetadata
    content_hash: str = Field(min_length=64, max_length=64)
    distance: float = Field(ge=0)
    dense_distance: Optional[float] = Field(default=None, ge=0)
    dense_rank: Optional[int] = Field(default=None, ge=1)
    bm25_score: Optional[float] = Field(default=None, ge=0)
    bm25_rank: Optional[int] = Field(default=None, ge=1)
    fusion_score: Optional[float] = Field(default=None, ge=0)
    fusion_rank: Optional[int] = Field(default=None, ge=1)
    reranker_score: Optional[float] = None
    reranker_rank: Optional[int] = Field(default=None, ge=1)
    final_rank: Optional[int] = Field(default=None, ge=1)
    metadata: Dict[str, str] = Field(default_factory=dict)


class RetrievalFilters(BaseModel):
    source_ids: List[str] = Field(default_factory=list)
    source_types: List[KnowledgeSourceType] = Field(default_factory=list)
    versions: List[str] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    return_candidate_pool: bool = False
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    mode: RetrievalMode = RetrievalMode.DENSE
    reranker: RerankerMode = RerankerMode.NONE
    dense_weight: float = Field(default=0.60, ge=0.0)
    bm25_weight: float = Field(default=0.40, ge=0.0)

    @model_validator(mode="after")
    def validate_hybrid_weights(self) -> "RetrievalRequest":
        if self.mode is RetrievalMode.HYBRID and self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("hybrid retrieval requires at least one positive weight")
        return self


class RetrievalStats(BaseModel):
    requested_top_k: int = Field(ge=1)
    mode: RetrievalMode = RetrievalMode.DENSE
    reranker: RerankerMode = RerankerMode.NONE
    raw_result_count: int = Field(ge=0)
    dense_result_count: int = Field(default=0, ge=0)
    bm25_result_count: int = Field(default=0, ge=0)
    deduplicated_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    reranked: bool = False


class RetrievalResult(BaseModel):
    query: str = Field(min_length=1)
    chunks: List[RetrievedKnowledgeChunk] = Field(default_factory=list)
    stats: RetrievalStats

    @property
    def citations(self) -> List[CitationMetadata]:
        return [chunk.citation for chunk in self.chunks]


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
