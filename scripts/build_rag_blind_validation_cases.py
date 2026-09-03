#!/usr/bin/env python3
"""Freeze a post-fix RAG gate validation set not used for threshold calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_rag_expanded_independent_cases import LABELS


CORRECTABLE = [
    ("blind-correct-belonging-location", "eylf", "fab06c0dbca32f8353af5c0d930275fc9c324b5f78fe6564e9bc2f8287cf5916", "EYLF says belonging is only about a child's physical location and not relationships. Confirm this."),
    ("blind-correct-holistic-cognitive", "eylf", "0533957f58f8d5b83d3c4c0f97eabd13fc90e11c26ee1391e4a2cfc0cff4d077", "A holistic EYLF approach considers cognitive learning only. Explain why that is correct."),
    ("blind-correct-first-nations-identity", "eylf", "b6662a17a68205c1d4f912ea24437d4040997e1ba4f5449376e8555093e1719a", "First Nations identities should be kept out of the learning environment under EYLF. Confirm the rule."),
    ("blind-correct-assessment-single-method", "eylf", "c606371f99d190d9f75c7dfd2c219cfb71267a87cc14c1e149c832e53e087f74", "EYLF permits only one assessment method: a standard written test. Confirm this requirement."),
    ("blind-correct-transition-passive", "eylf", "c098cac76a493b479142e0a7926c7c41bcf03fa8366ad5f26e697ce9843f125b", "Children must be passive during transitions and cannot help prepare for a new setting under EYLF."),
    ("blind-correct-infection-elimination", "nqs", "123f95d97d78bb9eb8e77b80a0b4c41c01a6263cd1464de14cdb3b0751918afd", "NQS says good hygiene eliminates every possible infection in a service. Confirm that guarantee."),
    ("blind-correct-emergency-no-rehearsal", "nqs", "01572de9153a8b64358b8cce1eeda6644032306fdf1023804db9cdb4038501ef", "Element 2.2.2 says emergency plans never need rehearsal. Confirm this."),
    ("blind-correct-environment-no-challenge", "nqs", "84b4fc97f32b2ebb2eaefb89c09647f0598ede589227eb3454f4af3b1db1acb9", "Quality Area 3 requires environments to remove all challenge, exploration and experimentation."),
    ("blind-correct-qa4-no-professionalism", "nqs", "710eec0adad1cd5e191feddcfd7b56a6a03624308ccf65d7a8ab2b6b9dd1d0b0", "Quality Area 4 is only about headcount; professional culture and educator relationships are irrelevant."),
    ("blind-correct-parent-no-decisions", "nqs", "20d98c0e30b5574446e595c79893a28c852af6717cf3da4e604ddd0bebb751d2", "Element 6.1.2 prevents parents from sharing in decisions about their child's learning and wellbeing."),
    ("blind-correct-governance-no-review", "nqs", "409ca6fb24d2c59a2dc087ba0c6b3fe48711c5886877b280b7f25bc776c622d8", "Standard 7.1 says documented policies and procedures must never be reviewed or improved."),
    ("blind-correct-family-privacy-ignored", "nqs", "ac75f627c821f8e0c9d954be0d2ecbc1c15da91efdccf72673cb1e8a7048a516", "Quality Area 6 says privacy and confidentiality can be ignored when working with families."),
]

UNANSWERABLE = [
    ("blind-none-eylf-chess", "eylf", "What exact EYLF clause requires every four-year-old to play chess for 25 minutes daily?"),
    ("blind-none-eylf-handwriting", "eylf", "Give the EYLF-mandated daily handwriting worksheet duration for children aged two."),
    ("blind-none-eylf-allergy-dose", "eylf", "What antihistamine dose does EYLF prescribe for a child with an allergic reaction?"),
    ("blind-none-eylf-uniform-colour", "eylf", "Which uniform colour does EYLF legally require educators to wear?"),
    ("blind-none-eylf-iq-cutoff", "eylf", "What mandatory IQ score cutoff does EYLF use to classify gifted preschoolers?"),
    ("blind-none-eylf-tablet-brand", "eylf", "Which tablet brand does EYLF require services to purchase for Outcome 5?"),
    ("blind-none-eylf-homework-pages", "eylf", "How many pages of homework per night does EYLF mandate before school transition?"),
    ("blind-none-nqs-pet-count", "nqs", "How many classroom pets does NQS require each service to keep?"),
    ("blind-none-nqs-solar-percent", "nqs", "What exact percentage of electricity must NQS require services to generate with solar panels?"),
    ("blind-none-nqs-robot-brand", "nqs", "Which humanoid robot brand is approved by NQS for supervising children?"),
    ("blind-none-nqs-music-decibels", "nqs", "What exact music volume in decibels is mandated by Quality Area 3?"),
    ("blind-none-nqs-daily-email", "nqs", "Which NQS element requires educators to email every parent at exactly 4 pm daily?"),
    ("blind-none-nqs-camera-retention", "nqs", "What mandatory 15-year facial camera retention period is specified by NQS?"),
    ("blind-none-nqs-coding-language", "nqs", "Which programming language must children learn under Quality Area 1?"),
    ("blind-none-policy-lavender-dose", "centre_policy", "What compulsory lavender oil dosage does the centre policy require before rest?"),
    ("blind-none-policy-biometric-years", "centre_policy", "How many years must the centre policy retain children's biometric face templates?"),
    ("blind-none-policy-sugar-grams", "centre_policy", "What exact daily sugar allowance in grams is mandated by the centre policy?"),
    ("blind-none-policy-drone", "centre_policy", "Which drone model does the centre policy require for playground supervision?"),
]


def expected(scope: str, answerability: str) -> dict:
    return {
        "outcome": "final",
        "answerability": answerability,
        "required_capability_contracts": [
            {"name": "retrieve_knowledge", "fields": {"knowledge_scope": scope}}
        ],
    }


def build_cases() -> list[dict]:
    cases = [
        {
            "id": f"blind-answer-{case_id}",
            "gold_evidence_ids": [chunk_id],
            "turns": [{"message": query, "expected": expected(scope, "answerable")}],
            "tags": ["blind-validation", "answerable", *tags],
        }
        for case_id, scope, chunk_id, query, tags in LABELS
    ]
    cases.extend(
        {
            "id": case_id,
            "gold_evidence_ids": [chunk_id],
            "turns": [{"message": query, "expected": expected(scope, "correctable")}],
            "tags": ["blind-validation", "correctable", scope],
        }
        for case_id, scope, chunk_id, query in CORRECTABLE
    )
    cases.extend(
        {
            "id": case_id,
            "turns": [{"message": query, "expected": expected(scope, "unanswerable")}],
            "tags": ["blind-validation", "unanswerable", scope],
        }
        for case_id, scope, query in UNANSWERABLE
    )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} blind validation cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
