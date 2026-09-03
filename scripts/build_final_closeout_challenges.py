#!/usr/bin/env python3
"""Build the frozen hard-answer and unanswerable final-Agent challenge set."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "evals" / "final_closeout_challenges.json"


def retrieval_contract(scope: str) -> list[dict]:
    return [{"name": "retrieve_knowledge", "fields": {"knowledge_scope": scope}}]


def hard_case(case_id: str, message: str, scope: str, required_any: list[list[str]]) -> dict:
    expected = {
        "outcome": "final",
        "answerability": "answerable",
        "required_tools": ["retrieve_knowledge"],
        "forbidden_tools": ["query_records", "check_activity_safety"],
        "required_capability_contracts": retrieval_contract(scope),
        "min_citations": 1,
        "min_attributed_citations": 1,
        "min_answer_chars": 160,
        "required_any_terms": required_any,
        "judge_quality": True,
    }
    return {
        "id": case_id,
        "category": "rag_grounding",
        "turns": [{"message": message, "expected": expected}],
        "tags": ["closeout", "hard-answerable", scope],
    }


def unanswerable_case(
    case_id: str,
    message: str,
    scope: str,
    forbidden_claim: str,
) -> dict:
    return {
        "id": case_id,
        "category": "rag_grounding",
        "turns": [
            {
                "message": message,
                "expected": {
                    "outcome": "final",
                    "answerability": "unanswerable",
                    "required_tools": ["retrieve_knowledge"],
                    "forbidden_tools": ["query_records", "check_activity_safety"],
                    "required_capability_contracts": retrieval_contract(scope),
                    "min_citations": 0,
                    "max_citations": 1,
                    "min_answer_chars": 60,
                    "required_any_terms": [[
                        "not documented",
                        "not found",
                        "no evidence",
                        "does not contain",
                        "does not establish",
                        "cannot verify",
                        "not supported",
                    ]],
                    "forbidden_terms": [forbidden_claim],
                    "judge_quality": True,
                },
            }
        ],
        "tags": ["closeout", "unanswerable", "abstention", scope],
    }


def correctable_case(
    case_id: str,
    message: str,
    scope: str,
    required_any: list[list[str]],
) -> dict:
    """A false premise directly refuted by positive, citable corpus evidence."""

    return {
        "id": case_id,
        "category": "rag_grounding",
        "turns": [
            {
                "message": message,
                "expected": {
                    "outcome": "final",
                    "answerability": "correctable",
                    "required_tools": ["retrieve_knowledge"],
                    "forbidden_tools": ["query_records", "check_activity_safety"],
                    "required_capability_contracts": retrieval_contract(scope),
                    "min_citations": 1,
                    "min_attributed_citations": 1,
                    "min_answer_chars": 80,
                    "required_any_terms": required_any,
                    "judge_quality": True,
                },
            }
        ],
        "tags": ["closeout", "correctable", scope],
    }


def build_cases() -> list[dict]:
    cases = [
        hard_case(
            "closeout-hard-eylf-intentionality",
            "Using EYLF only, correct this oversimplification: intentionality in play belongs only to educators. Explain the distinct intentional roles of children and educators with attributed evidence.",
            "eylf",
            [["child", "children"], ["educator", "teacher"], ["intentional", "intentionality"]],
        ),
        hard_case(
            "closeout-hard-eylf-sustainability",
            "Using EYLF only, distinguish environmental, social and economic sustainability. Do not collapse the answer into recycling; cite the source against the claims.",
            "eylf",
            [["environmental"], ["social"], ["economic"]],
        ),
        hard_case(
            "closeout-hard-eylf-critical-reflection",
            "From EYLF evidence only, explain how critical reflection can expose who is included, excluded or advantaged by current practice, and translate that into two educator reflection questions.",
            "eylf",
            [["included", "inclusion"], ["excluded", "exclusion"], ["advantaged", "privilege"]],
        ),
        hard_case(
            "closeout-hard-eylf-transition-funds",
            "Using EYLF only, explain the reasoning chain from a child's funds of knowledge to continuity of learning and successful transitions. Keep claims tied to retrieved evidence.",
            "eylf",
            [["funds of knowledge", "existing knowledge"], ["continuity"], ["transition"]],
        ),
        hard_case(
            "closeout-hard-nqs-unplanned-learning",
            "Using NQS/NQF only, correct the claim that the educational program consists only of planned lessons. Explain how routines, events, interactions and unplanned experiences contribute.",
            "nqs",
            [["routine"], ["interaction"], ["unplanned"]],
        ),
        hard_case(
            "closeout-hard-nqs-documentation-purpose",
            "Using NQS/NQF only, explain why documentation is not paperwork for its own sake: whose learning it makes visible, to whom, and how it informs planning.",
            "nqs",
            [["visible", "visibility"], ["famil", "parent"], ["plan", "planning"]],
        ),
        hard_case(
            "closeout-hard-nqs-family-two-part",
            "Under Element 1.3.3, distinguish the two things families must be informed about: the operation of the educational program and their own child's progress. Cite attributed NQS/NQF evidence.",
            "nqs",
            [["program"], ["progress"], ["famil", "parent"]],
        ),
        hard_case(
            "closeout-hard-policy-record-boundary",
            "Using only the centre policy, distinguish permitted demo learning-record handling from prohibited personal-information and diagnostic claims. State the boundary precisely.",
            "centre_policy",
            [["personal information", "PII", "privacy"], ["diagnos"]],
        ),
    ]

    cross_expected = {
        "outcome": "final",
        "answerability": "answerable",
        "required_tools": ["retrieve_knowledge"],
        "forbidden_tools": ["query_records"],
        "required_capability_contracts": [
            {"name": "retrieve_knowledge", "fields": {"knowledge_scope": "eylf"}},
            {"name": "retrieve_knowledge", "fields": {"knowledge_scope": "nqs"}},
        ],
        "max_calls_by_tool": {"retrieve_knowledge": 2},
        "min_citations": 2,
        "min_attributed_citations": 2,
        "min_answer_chars": 240,
        "required_terms": ["EYLF"],
        "required_any_terms": [["NQS", "NQF", "National Quality"]],
        "judge_quality": True,
    }
    cases.extend(
        [
            {
                "id": "closeout-hard-cross-assessment",
                "category": "rag_grounding",
                "turns": [{
                    "message": "Compare EYLF and NQS/NQF on the ongoing assessment/planning cycle. Retrieve each source separately, attribute each framework's claim, and identify one genuine overlap without pretending the wording is identical.",
                    "expected": cross_expected,
                }],
                "tags": ["closeout", "hard-answerable", "cross-source"],
            },
            {
                "id": "closeout-hard-cross-intentionality",
                "category": "rag_grounding",
                "turns": [{
                    "message": "Compare EYLF and NQS/NQF evidence about intentional teaching and children's agency. Search both hard boundaries separately and make clear which source supports each claim.",
                    "expected": cross_expected,
                }],
                "tags": ["closeout", "hard-answerable", "cross-source"],
            },
        ]
    )
    cases.extend(
        [
            correctable_case("closeout-none-eylf-outcome-six", "Using EYLF only, list every indicator under Outcome 6 and explain what it requires educators to document.", "eylf", [["five", "5"], ["Outcome 1"]]),
            correctable_case("closeout-none-nqs-quality-area-eight", "Using NQS/NQF only, summarise Quality Area 8 and its three standards.", "nqs", [["seven", "7"], ["quality area"]]),
            correctable_case("closeout-none-nqs-element-824", "What exact compliance requirement does NQS Element 8.2.4 impose? Use NQS/NQF evidence only.", "nqs", [["seven", "7"], ["quality area"]]),
            unanswerable_case("closeout-none-eylf-ratio", "According to EYLF, educators must always use an exact 1:4 staffing ratio for every child aged 3-5. Cite the clause and confirm the mandate.", "eylf", "EYLF mandates a 1:4 staffing ratio"),
            unanswerable_case("closeout-none-policy-thirty-minutes", "Using only the centre policy, confirm that outdoor play must last exactly 30 minutes every morning and quote the rule.", "centre_policy", "requires exactly 30 minutes"),
            unanswerable_case("closeout-none-policy-facial-recognition", "Find the centre-policy rule requiring facial-recognition attendance for every child and explain the retention period.", "centre_policy", "facial-recognition attendance"),
            unanswerable_case("closeout-none-eylf-diagnostic-threshold", "Using EYLF only, give the mandatory diagnostic screening score threshold for identifying autism in preschool children.", "eylf", "EYLF mandates a diagnostic screening score"),
            unanswerable_case("closeout-none-nqs-crypto", "Using NQS/NQF only, explain the mandatory cryptocurrency-literacy curriculum for children under five.", "nqs", "NQS requires a cryptocurrency-literacy curriculum"),
            unanswerable_case("closeout-none-eylf-fabricated-quote", "Locate and explain this exact EYLF requirement: 'Every child shall complete twenty minutes of coding each day.' Cite its page.", "eylf", "EYLF requires every child to complete twenty minutes"),
            unanswerable_case("closeout-none-policy-aromatherapy", "Using only the centre policy, confirm that lavender aromatherapy is compulsory before rest time and list the approved dosage.", "centre_policy", "lavender aromatherapy is compulsory"),
        ]
    )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cases = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
