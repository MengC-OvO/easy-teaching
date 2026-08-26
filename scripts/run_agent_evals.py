#!/usr/bin/env python3
"""Run the real-model, production-graph Agent evaluation suite."""

import argparse
import asyncio
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.agent_e2e import DEFAULT_CASES_PATH, format_report, run_agent_e2e  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the live EasyTeaching Main ReAct Agent.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "agent_e2e_report.json")
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument(
        "--include-long-term-memory",
        action="store_true",
        help="Use the real model-backed long-term-memory decision after each eligible case.",
    )
    parser.add_argument(
        "--quality-judge",
        action="store_true",
        help="Add one structured model quality judgment for each final-answer case.",
    )
    args = parser.parse_args()
    if not 0 <= args.min_pass_rate <= 1:
        parser.error("--min-pass-rate must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    report = asyncio.run(
        run_agent_e2e(
            cases_path=args.cases,
            case_ids=args.case_ids,
            input_cost_per_million=args.input_cost_per_million,
            output_cost_per_million=args.output_cost_per_million,
            include_long_term_memory=args.include_long_term_memory,
            quality_judge=args.quality_judge,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(format_report(report))
    print(f"report={args.output}")
    return 0 if report.summary.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
