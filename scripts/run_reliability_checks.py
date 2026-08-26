#!/usr/bin/env python3
"""Run the deterministic failure-injection matrix."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals import load_reliability_scenarios  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify EasyTeaching retry, fallback, stopping, and safe tracing."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the fault matrix without running it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = load_reliability_scenarios()
    print(f"EasyTeaching reliability matrix: {len(scenarios)} scenarios")
    for index, scenario in enumerate(scenarios, start=1):
        print(f"  {index:02d}. {scenario.id}: {scenario.expected}")
    if args.list:
        return 0

    print("\nRunning deterministic fault injections...\n")
    sys.stdout.flush()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *(scenario.test_node for scenario in scenarios),
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if completed.returncode == 0:
        print(f"\nReliability result: {len(scenarios)}/{len(scenarios)} passed")
    else:
        print("\nReliability result: one or more scenarios failed")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
