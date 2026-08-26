#!/usr/bin/env python3
"""Run the production API with a psycopg-compatible Windows event loop."""

import argparse
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asyncio_compat import run_async  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EasyTeaching API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = uvicorn.Config("app.main:app", host=args.host, port=args.port)
    server = uvicorn.Server(config)
    run_async(server.serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
