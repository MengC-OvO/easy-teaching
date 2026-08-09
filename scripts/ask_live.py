#!/usr/bin/env python3
"""Ask one question through the real EduFlow production execution path."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import build_api_runtime  # noqa: E402
from app.api.execution import execute_message  # noqa: E402
from app.config import settings  # noqa: E402
from app.schemas import StreamEventType  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one question through the real model, Main ReAct graph, "
            "PostgreSQL store, and PostgreSQL checkpoint saver."
        )
    )
    parser.add_argument("question", help="Teacher question to send to EduFlow.")
    parser.add_argument(
        "--teacher-id",
        default="terminal-demo-teacher",
        help="Synthetic teacher scope used by the demo.",
    )
    parser.add_argument(
        "--class-id",
        default="kangaroo-room",
        help="Synthetic class scope used by the demo.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the persisted graph trace after the answer.",
    )
    return parser.parse_args()


def validate_live_configuration() -> Optional[str]:
    if not settings.database_url or not settings.checkpoint_database_url:
        return "DATABASE_URL and CHECKPOINT_DATABASE_URL must be set in .env."
    if not settings.model_base_url or "your-model-provider.example" in settings.model_base_url:
        return "MODEL_BASE_URL must point to your real chat-completions provider."
    if not settings.model_api_key or settings.model_api_key.startswith("replace-with"):
        return "MODEL_API_KEY must contain a real local key in .env."
    return None


async def ask_live(args: argparse.Namespace) -> int:
    configuration_error = validate_live_configuration()
    if configuration_error:
        print(f"Configuration error: {configuration_error}", file=sys.stderr)
        return 2

    runtime = None
    session_id = str(uuid4())
    thread_id = str(uuid4())
    request_id = str(uuid4())
    try:
        runtime = await build_api_runtime()
        await runtime.store.create_conversation_session(
            session_id=session_id,
            thread_id=thread_id,
            teacher_id=args.teacher_id,
            class_id=args.class_id,
        )
        await runtime.store.create_conversation_run(
            request_id=request_id,
            session_id=session_id,
        )
        await runtime.store.append_conversation_event(
            request_id=request_id,
            session_id=session_id,
            event=StreamEventType.RUN_STARTED.value,
            data={"status": "accepted"},
        )

        print(f"Question: {args.question}")
        print("Running the real Main ReAct graph...\n")
        await execute_message(
            runtime=runtime,
            request_id=request_id,
            session_id=session_id,
            thread_id=thread_id,
            teacher_id=args.teacher_id,
            class_id=args.class_id,
            message=args.question,
        )

        run = await runtime.store.get_conversation_run(request_id)
        result = await runtime.store.get_conversation_run_result(request_id)
        status = run["status"] if run else "unknown"
        print(f"Status: {status}")
        if result is None:
            print("No draft was produced. Check the persisted failed event or server logs.")
            return 1

        draft = result["draft"]
        print(f"\n{draft.get('title') or 'EduFlow answer'}")
        print("=" * 60)
        print(draft.get("content", ""))

        citations = result.get("citations", [])
        if citations:
            print("\nCitations")
            print("-" * 60)
            for index, citation in enumerate(citations, start=1):
                label = citation.get("title") or citation.get("source") or "Unknown source"
                location = citation.get("url") or citation.get("section") or ""
                print(f"{index}. {label}{f' — {location}' if location else ''}")

        if args.trace:
            events = await runtime.store.list_conversation_events(request_id=request_id)
            trace_events = [item for item in events if item["event"] == "trace"]
            print("\nExecution trace")
            print("-" * 60)
            for event in trace_events:
                data = event["data"]
                metadata = data.get("metadata", {})
                code = metadata.get("code") if isinstance(metadata, dict) else None
                print(f"{event['sequence']:02d}. {data.get('step', 'trace')}: "
                      f"{data.get('message', '')}{f' [{code}]' if code else ''}")

        print(f"\nSession: {session_id}")
        print(f"Request: {request_id}")
        return 0 if status == "completed" else 1
    except Exception as error:
        detail = str(error)
        if "greenlet" in detail.lower():
            detail = "SQLAlchemy asyncio support is missing; run pip install -r requirements.txt."
        print(
            "Live run could not start. Confirm PostgreSQL is running, migrations "
            f"are applied, and .env is configured ({type(error).__name__}: {detail}).",
            file=sys.stderr,
        )
        return 2
    finally:
        if runtime is not None:
            await runtime.close()


def main() -> int:
    return asyncio.run(ask_live(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
