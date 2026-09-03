"""Deterministic score gate for retrieved policy/framework evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.schemas import KnowledgeSourceType, RetrievedKnowledgeChunk, RerankerMode


@dataclass(frozen=True)
class EvidenceGateDecision:
    answerable: bool
    reason: str
    supported_chunks: List[RetrievedKnowledgeChunk]


# Frozen against the public answerability calibration set, then checked without
# retuning on blind and robustness sets. BM25 scales differ by corpus, so
# source-specific lexical thresholds are intentional. RRF/fusion scores are not
# used because they encode rank rather than absolute relevance.
EYLF_MAX_DENSE_DISTANCE = 0.31
EYLF_STRONG_DENSE_DISTANCE = 0.22
EYLF_MIN_BM25_SCORE = 13.0
NQS_STRONG_DENSE_DISTANCE = 0.22
NQS_MAX_DENSE_DISTANCE = 0.30
NQS_MIN_BM25_SCORE = 15.0
CENTRE_MAX_DENSE_DISTANCE = 0.30
CENTRE_MIN_BM25_SCORE = 22.0
SCOPED_MAX_DENSE_DISTANCE = 0.35
DEFAULT_MAX_DENSE_DISTANCE = 0.30
DEFAULT_MIN_BM25_SCORE = 15.0
DEEP_MIN_RERANKER_SCORE = -4.40


def assess_retrieved_evidence(
    chunks: List[RetrievedKnowledgeChunk],
    *,
    reranker: RerankerMode,
) -> EvidenceGateDecision:
    """Keep only chunks that clear calibrated absolute relevance thresholds."""

    if not chunks:
        return EvidenceGateDecision(False, "no_evidence_retrieved", [])

    supported = [
        chunk
        for chunk in chunks
        if _passes_base_gate(chunk) and _passes_deep_gate(chunk, reranker)
    ]
    if not supported:
        return EvidenceGateDecision(False, "evidence_below_relevance_threshold", [])
    return EvidenceGateDecision(True, "relevance_threshold_passed", supported)


def _passes_base_gate(chunk: RetrievedKnowledgeChunk) -> bool:
    dense = chunk.dense_distance
    bm25 = chunk.bm25_score
    if dense is None or bm25 is None:
        return False

    citation = chunk.citation
    if "scoped_hybrid_score" in chunk.metadata:
        return dense <= SCOPED_MAX_DENSE_DISTANCE and bm25 > 0
    if citation.source_id == "eylf-v2":
        return dense <= EYLF_STRONG_DENSE_DISTANCE or (
            dense <= EYLF_MAX_DENSE_DISTANCE and bm25 >= EYLF_MIN_BM25_SCORE
        )
    if citation.source_id == "nqs-guide-qa1":
        return dense <= NQS_STRONG_DENSE_DISTANCE or (
            dense <= NQS_MAX_DENSE_DISTANCE and bm25 >= NQS_MIN_BM25_SCORE
        )
    if citation.source_type in {
        KnowledgeSourceType.CENTRE,
        KnowledgeSourceType.SYNTHETIC,
    }:
        return (
            dense <= CENTRE_MAX_DENSE_DISTANCE
            and bm25 >= CENTRE_MIN_BM25_SCORE
        )
    return dense <= DEFAULT_MAX_DENSE_DISTANCE and bm25 >= DEFAULT_MIN_BM25_SCORE


def _passes_deep_gate(
    chunk: RetrievedKnowledgeChunk,
    reranker: RerankerMode,
) -> bool:
    if reranker is RerankerMode.NONE:
        return True
    return (
        chunk.reranker_score is not None
        and chunk.reranker_score >= DEEP_MIN_RERANKER_SCORE
    )
