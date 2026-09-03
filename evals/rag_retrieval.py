"""Ground-truth retrieval metrics for the local RAG pipeline."""

import math
from statistics import mean
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from pydantic import BaseModel, Field, model_validator

from app.schemas import KnowledgeChunk, KnowledgeScope, RetrievalResult, source_ids_for_scope


class RagGoldEvidence(BaseModel):
    source_id: str = Field(min_length=1)
    chunk_id: Optional[str] = Field(default=None, min_length=1)
    page: Optional[int] = Field(default=None, ge=1)
    section_contains: Optional[str] = Field(default=None, min_length=1)
    relevance: int = Field(default=3, ge=1, le=3)

    @model_validator(mode="after")
    def validate_locator(self) -> "RagGoldEvidence":
        if self.chunk_id is None and self.page is None and self.section_contains is None:
            raise ValueError("gold evidence needs a chunk, page, or section locator")
        return self


class RagRetrievalCase(BaseModel):
    case_id: str = Field(min_length=1)
    split: Literal["dev", "test"] = "dev"
    query: str = Field(min_length=2)
    scope: KnowledgeScope
    tags: List[str] = Field(default_factory=list)
    relevant_evidence: List[RagGoldEvidence] = Field(min_length=1)


class RagRetrievedEvidence(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str
    source_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    relevance: int = Field(ge=0, le=3)


class RagCaseMetrics(BaseModel):
    case_id: str
    split: Literal["dev", "test"]
    mode: str
    latency_ms: float = Field(ge=0)
    recall_at_k: Dict[int, float]
    hit_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    reciprocal_rank: float = Field(ge=0, le=1)
    average_precision: float = Field(ge=0, le=1)
    ndcg_at_k: Dict[int, float]
    scope_violation_count: int = Field(ge=0)
    returned_count: int = Field(ge=0)
    citation_correct_count: int = Field(ge=0)
    retrieved: List[RagRetrievedEvidence]


class RagModeSummary(BaseModel):
    mode: str
    case_count: int = Field(ge=1)
    recall_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float = Field(ge=0, le=1)
    map: float = Field(ge=0, le=1)
    ndcg_at_k: Dict[int, float]
    scope_violation_rate: float = Field(ge=0, le=1)
    citation_correctness: float = Field(ge=0, le=1)
    latency_p50_ms: float = Field(ge=0)
    latency_p95_ms: float = Field(ge=0)


def evaluate_case(
    *,
    case: RagRetrievalCase,
    mode: str,
    result: RetrievalResult,
    latency_ms: float,
    ks: Sequence[int],
    chunk_catalog: Mapping[str, KnowledgeChunk],
) -> RagCaseMetrics:
    grades = _ranked_relevance_grades(case, result)
    allowed_source_ids = set(source_ids_for_scope(case.scope))
    scope_violations = sum(
        1
        for chunk in result.chunks
        if allowed_source_ids and chunk.citation.source_id not in allowed_source_ids
    )
    citation_correct_count = sum(
        1 for chunk in result.chunks if citation_is_correct(chunk, chunk_catalog)
    )
    retrieved = [
        RagRetrievedEvidence(
            rank=rank,
            chunk_id=chunk.chunk_id,
            source_id=chunk.citation.source_id,
            page=chunk.citation.page,
            section=chunk.citation.section,
            relevance=grades[rank - 1],
        )
        for rank, chunk in enumerate(result.chunks, start=1)
    ]
    first_relevant_rank = next(
        (rank for rank, grade in enumerate(grades, start=1) if grade > 0), None
    )
    return RagCaseMetrics(
        case_id=case.case_id,
        split=case.split,
        mode=mode,
        latency_ms=latency_ms,
        recall_at_k={k: _recall_at_k(grades, len(case.relevant_evidence), k) for k in ks},
        hit_at_k={k: float(any(grade > 0 for grade in grades[:k])) for k in ks},
        precision_at_k={k: _precision_at_k(grades, k) for k in ks},
        reciprocal_rank=(1 / first_relevant_rank if first_relevant_rank else 0.0),
        average_precision=_average_precision(grades, len(case.relevant_evidence)),
        ndcg_at_k={
            k: _ndcg_at_k(
                grades,
                [evidence.relevance for evidence in case.relevant_evidence],
                k,
            )
            for k in ks
        },
        scope_violation_count=scope_violations,
        returned_count=len(result.chunks),
        citation_correct_count=citation_correct_count,
        retrieved=retrieved,
    )


def summarize_mode(results: Sequence[RagCaseMetrics], ks: Sequence[int]) -> RagModeSummary:
    if not results:
        raise ValueError("results must not be empty")
    returned = sum(result.returned_count for result in results)
    latencies = [result.latency_ms for result in results]
    return RagModeSummary(
        mode=results[0].mode,
        case_count=len(results),
        recall_at_k={k: mean(result.recall_at_k[k] for result in results) for k in ks},
        hit_rate_at_k={k: mean(result.hit_at_k[k] for result in results) for k in ks},
        precision_at_k={k: mean(result.precision_at_k[k] for result in results) for k in ks},
        mrr=mean(result.reciprocal_rank for result in results),
        map=mean(result.average_precision for result in results),
        ndcg_at_k={k: mean(result.ndcg_at_k[k] for result in results) for k in ks},
        scope_violation_rate=(
            sum(result.scope_violation_count for result in results) / returned
            if returned
            else 0.0
        ),
        citation_correctness=(
            sum(result.citation_correct_count for result in results) / returned
            if returned
            else 0.0
        ),
        latency_p50_ms=percentile(latencies, 0.50),
        latency_p95_ms=percentile(latencies, 0.95),
    )


def citation_is_correct(retrieved, chunk_catalog: Mapping[str, KnowledgeChunk]) -> bool:
    stored = chunk_catalog.get(retrieved.chunk_id)
    if stored is None:
        return False
    citation = retrieved.citation
    expected = stored.citation
    return (
        retrieved.content == stored.content
        and retrieved.content_hash == stored.content_hash
        and citation.source_id == expected.source_id
        and citation.source_type == expected.source_type
        and citation.title == expected.title
        and citation.version == expected.version
        and citation.section == expected.section
        and citation.page == expected.page
        and citation.uri == expected.uri
    )


def percentile(values: Iterable[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def _ranked_relevance_grades(case: RagRetrievalCase, result: RetrievalResult) -> List[int]:
    unmatched = set(range(len(case.relevant_evidence)))
    grades: List[int] = []
    for chunk in result.chunks:
        matches = [
            index
            for index in unmatched
            if _matches_gold(chunk, case.relevant_evidence[index])
        ]
        if not matches:
            grades.append(0)
            continue
        best = max(matches, key=lambda index: case.relevant_evidence[index].relevance)
        unmatched.remove(best)
        grades.append(case.relevant_evidence[best].relevance)
    return grades


def _matches_gold(chunk, gold: RagGoldEvidence) -> bool:
    citation = chunk.citation
    if citation.source_id != gold.source_id:
        return False
    if gold.chunk_id is not None and chunk.chunk_id != gold.chunk_id:
        return False
    if gold.page is not None and citation.page != gold.page:
        return False
    if gold.section_contains is not None:
        actual_section = " ".join((citation.section or "").lower().split())
        expected_section = " ".join(gold.section_contains.lower().split())
        if expected_section not in actual_section:
            return False
    return True


def _recall_at_k(grades: Sequence[int], relevant_count: int, k: int) -> float:
    return sum(1 for grade in grades[:k] if grade > 0) / relevant_count


def _precision_at_k(grades: Sequence[int], k: int) -> float:
    return sum(1 for grade in grades[:k] if grade > 0) / k


def _average_precision(grades: Sequence[int], relevant_count: int) -> float:
    if relevant_count <= 0:
        return 0.0
    precision_sum = 0.0
    relevant_seen = 0
    for rank, grade in enumerate(grades, start=1):
        if grade <= 0:
            continue
        relevant_seen += 1
        precision_sum += relevant_seen / rank
    return precision_sum / relevant_count


def _ndcg_at_k(grades: Sequence[int], ideal_grades: Sequence[int], k: int) -> float:
    actual = _dcg(grades[:k])
    ideal = _dcg(sorted(ideal_grades, reverse=True)[:k])
    return actual / ideal if ideal else 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )
