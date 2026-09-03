#!/usr/bin/env python3
"""Run the production deterministic RAG gate on labelled answerability cases."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import KnowledgeScope
from app.tools.controlled_tools.knowledge_search import (
    RetrieveKnowledgeInput,
    build_retrieve_knowledge_tool,
)


def load_case(case: dict) -> tuple[str, str, KnowledgeScope]:
    turn = case["turns"][0]
    expected = turn["expected"]
    scope = KnowledgeScope(
        expected["required_capability_contracts"][0]["fields"]["knowledge_scope"]
    )
    return turn["message"], expected["answerability"], scope


async def evaluate(cases: list[dict], modes: list[str]) -> dict:
    tool = build_retrieve_knowledge_tool()
    results = []
    total = len(cases) * len(modes)
    completed = 0
    for case in cases:
        query, expected, scope = load_case(case)
        for mode in modes:
            completed += 1
            output = await tool.async_handler(
                RetrieveKnowledgeInput(
                    query=query,
                    knowledge_scope=scope,
                    mode=mode,
                    top_k=3 if mode == "standard" else 5,
                )
            )
            actual = str(output.data["answerability"])
            passed = (
                actual == "answerable"
                if expected in {"answerable", "correctable"}
                else actual == "insufficient"
            )
            results.append(
                {
                    "case_id": case["id"],
                    "mode": mode,
                    "expected_answerability": expected,
                    "actual_answerability": actual,
                    "passed": passed,
                    "answerability_reason": output.data["answerability_reason"],
                    "retrieved_count": output.data["retrieved_count"],
                    "returned_count": output.data["returned_count"],
                }
            )
            print(
                f"[{completed:03d}/{total:03d}] {mode:<8} {case['id']:<42} "
                f"expected={expected:<12} actual={actual:<12} {'PASS' if passed else 'FAIL'}"
            )

    summaries = {}
    for mode in modes:
        selected = [item for item in results if item["mode"] == mode]
        answerable = [item for item in selected if item["expected_answerability"] == "answerable"]
        correctable = [item for item in selected if item["expected_answerability"] == "correctable"]
        unanswerable = [item for item in selected if item["expected_answerability"] == "unanswerable"]
        summaries[mode] = {
            "case_count": len(selected),
            "answerable_allow_rate": rate(answerable),
            "correctable_allow_rate": rate(correctable),
            "unanswerable_reject_rate": rate(unanswerable),
            **classification_metrics(selected),
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "production retrieve_knowledge path plus deterministic evidence gate",
        "summaries": summaries,
        "results": results,
    }


def rate(items: list[dict]) -> float | None:
    if not items:
        return None
    return sum(bool(item["passed"]) for item in items) / len(items)


def classification_metrics(items: list[dict]) -> dict:
    """Treat gate allow as positive and report the full binary confusion matrix."""

    tp = sum(
        item["expected_answerability"] in {"answerable", "correctable"}
        and item["actual_answerability"] == "answerable"
        for item in items
    )
    fn = sum(
        item["expected_answerability"] in {"answerable", "correctable"}
        and item["actual_answerability"] == "insufficient"
        for item in items
    )
    fp = sum(
        item["expected_answerability"] == "unanswerable"
        and item["actual_answerability"] == "answerable"
        for item in items
    )
    tn = sum(
        item["expected_answerability"] == "unanswerable"
        and item["actual_answerability"] == "insufficient"
        for item in items
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "allow_precision": precision,
        "allow_recall": recall,
        "allow_f1": f1,
        "reject_specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "false_reject_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_allow_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["standard", "deep"],
        default=["standard", "deep"],
    )
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    payload = asyncio.run(evaluate(cases, list(dict.fromkeys(args.modes))))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summaries"], indent=2))
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
