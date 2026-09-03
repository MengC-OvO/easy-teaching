import json
from collections import Counter
from pathlib import Path

from scripts.build_rag_expanded_answerability_cases import (
    ANSWERABLE,
    CORRECTABLE,
    UNANSWERABLE,
)
from scripts.build_rag_expanded_independent_cases import LABELS
from scripts.build_rag_expanded_robustness_cases import (
    adjacent_distractor,
    telegraphic,
    typo_noise,
)
from scripts.build_rag_blind_validation_cases import build_cases as build_blind_cases


ROOT = Path(__file__).resolve().parents[1]


def test_expanded_retrieval_labels_are_independent_real_corpus_chunks() -> None:
    chunks = {
        row["chunk_id"]: row
        for line in (ROOT / "data/knowledge/processed/chunks.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    original_gold = {
        evidence["chunk_id"]
        for case in json.loads(
            (ROOT / "data/evals/rag_final_cases.json").read_text(encoding="utf-8")
        )
        for evidence in case["relevant_evidence"]
    }
    ids = [case_id for case_id, *_ in LABELS]
    gold = [chunk_id for _, _, chunk_id, _, _ in LABELS]
    scopes = Counter(scope for _, scope, *_ in LABELS)

    assert len(LABELS) == 54
    assert len(ids) == len(set(ids))
    assert len(gold) == len(set(gold))
    assert scopes == {"eylf": 20, "nqs": 34}
    assert set(gold) <= chunks.keys()
    assert set(gold).isdisjoint(original_gold)


def test_expanded_answerability_set_has_three_distinct_decision_classes() -> None:
    ordinary_ids = {case_id for case_id, *_ in ANSWERABLE}
    challenge_ids = {case_id for case_id, *_ in UNANSWERABLE}

    assert len(ANSWERABLE) == 20
    assert len(CORRECTABLE) == 14
    assert len(UNANSWERABLE) - len(CORRECTABLE) == 6
    assert ordinary_ids.isdisjoint(challenge_ids)
    assert set(CORRECTABLE) <= challenge_ids
    assert len(ordinary_ids | challenge_ids) == 40


def test_every_independent_question_has_three_nonidentical_robustness_forms() -> None:
    transforms = (typo_noise, telegraphic, adjacent_distractor)
    variants = []
    for case_id, _, _, query, _ in LABELS:
        generated = [transform(query) for transform in transforms]
        assert len(set(generated)) == 3, case_id
        assert all(value != query for value in generated), case_id
        variants.extend((case_id, value) for value in generated)

    assert len(variants) == 162


def test_expanded_labels_cover_all_nqs_quality_areas_after_qa1() -> None:
    tags = {tag for *_, case_tags in LABELS for tag in case_tags}

    assert {"qa2", "qa3", "qa4", "qa5", "qa6", "qa7"} <= tags
    assert {"planning-cycle", "assessment", "first-nations", "operational"} <= tags


def test_blind_gate_validation_has_broad_decision_and_scope_coverage() -> None:
    cases = build_blind_cases()
    decisions = Counter(
        case["turns"][0]["expected"]["answerability"] for case in cases
    )
    scopes = Counter(
        case["turns"][0]["expected"]["required_capability_contracts"][0]["fields"]["knowledge_scope"]
        for case in cases
    )

    assert len(cases) == 84
    assert decisions == {"answerable": 54, "unanswerable": 18, "correctable": 12}
    assert set(scopes) == {"eylf", "nqs", "centre_policy"}
