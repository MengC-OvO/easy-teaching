import hashlib
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class KnowledgeSourceType(str, Enum):
    OFFICIAL = "official"
    SYNTHETIC = "synthetic"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


class RerankerMode(str, Enum):
    NONE = "none"
    LEXICAL = "lexical"
    CROSS_ENCODER = "cross_encoder"


class PolicyRAGStatus(str, Enum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    REFUSED = "refused"
    EVIDENCE_CONFLICT = "evidence_conflict"


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
    metadata: Dict[str, str] = Field(default_factory=dict)


class RetrievalFilters(BaseModel):
    source_ids: List[str] = Field(default_factory=list)
    source_types: List[KnowledgeSourceType] = Field(default_factory=list)
    versions: List[str] = Field(default_factory=list)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    mode: RetrievalMode = RetrievalMode.DENSE
    reranker: RerankerMode = RerankerMode.NONE
    use_reranker: bool = False
    dense_weight: float = Field(default=0.65, ge=0.0)
    bm25_weight: float = Field(default=0.35, ge=0.0)

    @model_validator(mode="after")
    def sync_legacy_reranker_flag(self) -> "RetrievalRequest":
        if self.use_reranker and self.reranker is RerankerMode.NONE:
            self.reranker = RerankerMode.LEXICAL
        self.use_reranker = self.reranker is not RerankerMode.NONE
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


class PolicyEvidence(BaseModel):
    evidence_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    citation: CitationMetadata
    relevance_distance: float = Field(ge=0)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_retrieved_chunk(
        cls,
        chunk: RetrievedKnowledgeChunk,
        *,
        index: int,
    ) -> "PolicyEvidence":
        return cls(
            evidence_id=f"E{index}",
            content=chunk.content,
            citation=chunk.citation,
            relevance_distance=chunk.distance,
            metadata=chunk.metadata,
        )


class PolicyRAGResult(BaseModel):
    status: PolicyRAGStatus
    question: str = Field(min_length=1)
    answer: Optional[str] = None
    evidence: List[PolicyEvidence] = Field(default_factory=list)
    citations: List[CitationMetadata] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    refusal_reason: Optional[str] = None
    retrieval: RetrievalResult

    @model_validator(mode="after")
    def validate_status_payload(self) -> "PolicyRAGResult":
        if self.status is PolicyRAGStatus.ANSWERED and not self.answer:
            raise ValueError("answered policy result must include an answer")
        if self.status is PolicyRAGStatus.NEEDS_CLARIFICATION and not self.clarification_question:
            raise ValueError("clarification policy result must include a question")
        if self.status in {PolicyRAGStatus.REFUSED, PolicyRAGStatus.EVIDENCE_CONFLICT}:
            if not self.refusal_reason:
                raise ValueError("refused or conflicting policy result must include a reason")
        return self


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
