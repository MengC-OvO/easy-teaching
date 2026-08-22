#!/usr/bin/env python3
"""Run the Week 4 evaluation suite without starting the FastAPI server."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals import EvalCategory, EvalMode, run_eval_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate EasyTeaching routing, tools, RAG, memory, safety, and graph paths."
    )
    parser.add_argument(
        "--live-model",
        action="store_true",
        help="Use the configured real model for routing and grounded RAG answers.",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=[category.value for category in EvalCategory],
        help="Run one category; repeat this option to select several.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally save the complete JSON report to this path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete JSON report instead of the concise summary.",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Exit with status 1 when pass rate is below this 0-1 threshold.",
    )
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")
    if args.input_cost_per_million < 0 or args.output_cost_per_million < 0:
        parser.error("token prices cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    mode = EvalMode.LIVE_MODEL if args.live_model else EvalMode.OFFLINE
    categories = (
        [EvalCategory(value) for value in args.category]
        if args.category
        else None
    )
    report = run_eval_suite(
        mode=mode,
        categories=categories,
        input_cost_per_million=args.input_cost_per_million,
        output_cost_per_million=args.output_cost_per_million,
    )
    report_json = report.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json + "\n", encoding="utf-8")
    if args.json:
        print(report_json)
    else:
        summary = report.summary
        print(
            f"EasyTeaching eval ({report.mode.value}): "
            f"{summary.passed}/{summary.total} passed "
            f"({summary.pass_rate:.1%}), average score {summary.average_score:.3f}"
        )
        for category, result in summary.categories.items():
            print(
                f"  {category.value:10} "
                f"{result.passed}/{result.total} passed "
                f"({result.pass_rate:.1%})"
            )
        failed = [result for result in report.results if not result.passed]
        if failed:
            print("Failed cases:")
            for result in failed:
                failed_checks = ", ".join(
                    check.name for check in result.checks if not check.passed
                )
                print(f"  {result.case_id}: {failed_checks}")
        print(
            f"Model calls: {summary.token_usage.model_calls}; "
            f"tokens: {summary.token_usage.total_tokens}; "
            f"estimated cost: ${summary.estimated_cost_usd:.6f}"
        )
        if args.output:
            print(f"Report saved to: {args.output}")
    return 0 if report.summary.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
