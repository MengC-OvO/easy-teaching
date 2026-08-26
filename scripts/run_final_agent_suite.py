#!/usr/bin/env python3
"""Run the independent final production-path Agent evaluation."""

import argparse
import asyncio
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.final_agent_suite import (  # noqa: E402
    DEFAULT_FINAL_CASES_PATH,
    format_final_report,
    run_final_agent_suite,
)


def main() -> int:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    logging.basicConfig(level=logging.WARNING, force=True)
    for name in ("httpx", "httpcore", "chromadb", "sqlalchemy.engine"):
        logging.getLogger(name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="Run the final EasyTeaching Agent suite.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_FINAL_CASES_PATH)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "final_agent_evaluation.json",
    )
    args = parser.parse_args()
    report = asyncio.run(
        run_final_agent_suite(
            cases_path=args.cases,
            concurrency=args.concurrency,
            case_ids=args.case_ids,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(format_final_report(report))
    print(f"report={args.output}")
    return 0 if report.summary.release_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
