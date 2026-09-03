#!/usr/bin/env python3
"""Reapply deterministic checks to preserved model/judge outputs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_public_rag_answerability_evals import RagAnswer, RagJudge, deterministic_checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = {
        case["id"]: case
        for case in json.loads(args.cases.read_text(encoding="utf-8"))
    }
    report = json.loads(args.report.read_text(encoding="utf-8"))
    for result in report["results"]:
        case = cases[result["case_id"]]
        answer = RagAnswer.model_validate(result["answer"])
        judge = RagJudge.model_validate(result["judge"])
        result["expected_answerability"] = case["turns"][0]["expected"][
            "answerability"
        ]
        valid_ids = set(result.get("retrieved_evidence_ids") or [])
        if not valid_ids and next(
            check for check in result["checks"]
            if check["name"] == "citation_ids_valid"
        )["passed"]:
            valid_ids = set(answer.cited_evidence_ids)
        result["checks"] = deterministic_checks(case, answer, judge, valid_ids)
        result["passed"] = all(check["passed"] for check in result["checks"])
    answerable = [item for item in report["results"] if item["expected_answerability"] == "answerable"]
    correctable = [item for item in report["results"] if item["expected_answerability"] == "correctable"]
    unanswerable = [item for item in report["results"] if item["expected_answerability"] == "unanswerable"]
    report["summary"].update(
        {
            "passed": sum(item["passed"] for item in report["results"]),
            "pass_rate": sum(item["passed"] for item in report["results"]) / len(report["results"]),
            "answerable_pass_rate": sum(item["passed"] for item in answerable) / len(answerable),
            "correction_pass_rate": (
                sum(item["passed"] for item in correctable) / len(correctable)
                if correctable else None
            ),
            "abstention_pass_rate": sum(item["passed"] for item in unanswerable) / len(unanswerable),
        }
    )
    report["deterministic_recheck"] = {
        "source_report": str(args.report),
        "reason": "forbidden phrases narrowed from raw topic substrings to affirmative fabricated claims",
        "model_calls_reused": True,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["pass_rate"] >= 0.85 else 1


if __name__ == "__main__":
    raise SystemExit(main())
