from app.schemas import (
    CitationMetadata,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    RerankerMode,
)
from app.services.evidence_gate import assess_retrieved_evidence
from scripts.run_rag_gate_evals import classification_metrics


def chunk(
    *,
    source_id: str,
    source_type: KnowledgeSourceType,
    dense: float,
    bm25: float,
    reranker: float | None = None,
) -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        chunk_id=f"{source_id}-{dense}-{bm25}",
        content="Calibrated evidence.",
        citation=CitationMetadata(
            source_id=source_id,
            source_type=source_type,
            title=source_id,
            version="1",
        ),
        content_hash="a" * 64,
        distance=dense,
        dense_distance=dense,
        bm25_score=bm25,
        reranker_score=reranker,
    )


def test_standard_gate_uses_source_specific_absolute_thresholds() -> None:
    eylf = chunk(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        dense=0.29,
        bm25=13.0,
    )
    nqs_strong_dense = chunk(
        source_id="nqs-guide-qa1",
        source_type=KnowledgeSourceType.OFFICIAL,
        dense=0.21,
        bm25=1.0,
    )
    centre = chunk(
        source_id="synthetic-centre-policies",
        source_type=KnowledgeSourceType.SYNTHETIC,
        dense=0.29,
        bm25=22.0,
    )

    decision = assess_retrieved_evidence(
        [eylf, nqs_strong_dense, centre],
        reranker=RerankerMode.NONE,
    )

    assert decision.answerable is True
    assert decision.supported_chunks == [eylf, nqs_strong_dense, centre]


def test_standard_gate_rejects_nearest_but_low_confidence_chunks() -> None:
    weak = chunk(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        dense=0.31,
        bm25=12.9,
    )

    decision = assess_retrieved_evidence([weak], reranker=RerankerMode.NONE)

    assert decision.answerable is False
    assert decision.reason == "evidence_below_relevance_threshold"
    assert decision.supported_chunks == []


def test_eylf_gate_accepts_high_lexical_evidence_just_over_point_three() -> None:
    evidence = chunk(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        dense=0.305,
        bm25=14.0,
    )

    decision = assess_retrieved_evidence([evidence], reranker=RerankerMode.NONE)

    assert decision.answerable is True


def test_deep_gate_adds_cross_encoder_threshold() -> None:
    weak_reranker = chunk(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        dense=0.20,
        bm25=20.0,
        reranker=-4.41,
    )
    passing = weak_reranker.model_copy(
        update={"chunk_id": "passing", "reranker_score": -4.40}
    )

    rejected = assess_retrieved_evidence(
        [weak_reranker],
        reranker=RerankerMode.CROSS_ENCODER,
    )
    accepted = assess_retrieved_evidence(
        [passing],
        reranker=RerankerMode.CROSS_ENCODER,
    )

    assert rejected.answerable is False
    assert accepted.answerable is True
    assert accepted.supported_chunks == [passing]


def test_scoped_local_gate_uses_rank_agreement_not_public_bm25_scale() -> None:
    local = chunk(
        source_id="teacher-upload",
        source_type=KnowledgeSourceType.CENTRE,
        dense=0.30,
        bm25=0.1,
    )
    local.metadata = {"scoped_hybrid_score": "0.016393"}

    decision = assess_retrieved_evidence([local], reranker=RerankerMode.NONE)

    assert decision.answerable is True


def test_gate_fails_closed_when_absolute_scores_are_missing() -> None:
    item = chunk(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        dense=0.20,
        bm25=20.0,
    )
    item.dense_distance = None

    decision = assess_retrieved_evidence([item], reranker=RerankerMode.NONE)

    assert decision.answerable is False


def test_gate_eval_reports_full_confusion_metrics() -> None:
    items = [
        {"expected_answerability": "answerable", "actual_answerability": "answerable"},
        {"expected_answerability": "correctable", "actual_answerability": "insufficient"},
        {"expected_answerability": "unanswerable", "actual_answerability": "answerable"},
        {"expected_answerability": "unanswerable", "actual_answerability": "insufficient"},
    ]

    metrics = classification_metrics(items)

    assert metrics["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert metrics["allow_precision"] == 0.5
    assert metrics["allow_recall"] == 0.5
    assert metrics["allow_f1"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
