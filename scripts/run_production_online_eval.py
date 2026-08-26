#!/usr/bin/env python3
"""Run the synthetic-data production-path online evaluation."""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.production_online import (  # noqa: E402
    format_production_report,
    run_production_online_eval,
)


def main() -> int:
    if sys.platform == "win32":
        # psycopg async connections do not support Windows' Proactor loop.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Online providers can emit entire request/response payloads at INFO level.
    # Keep the CLI readable and avoid leaking synthetic prompts into CI logs.
    logging.basicConfig(level=logging.WARNING, force=True)
    for logger_name in ("httpx", "httpcore", "chromadb", "sqlalchemy.engine"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "production_online_report.json",
    )
    parser.add_argument("--p95-slo-ms", type=float, default=30_000)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help=(
            "Run only a named production group; repeat as needed. Groups: "
            "health-validation, activity, policy-rag, observation-write-query, "
            "versioned-reference, cross-framework, idempotency, concurrency."
        ),
    )
    args = parser.parse_args()

    print("production_online_eval=starting", flush=True)
    try:
        report = run_production_online_eval(
            p95_slo_ms=args.p95_slo_ms,
            concurrency=args.concurrency,
            case_ids=args.case_ids,
        )
    except Exception:
        print("production_online_eval=crashed", file=sys.stderr, flush=True)
        traceback.print_exc(limit=20)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(format_production_report(report))
    print(f"report={args.output}")
    return 0 if report.summary.slo_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
