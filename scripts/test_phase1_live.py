#!/usr/bin/env python3
"""User-run live checks for the phase-one Agent and tool architecture.

This script intentionally does not auto-approve writes. It calls the configured
chat and embedding providers, so Codex must not run it during offline checks.
"""

import argparse
import sys
from pathlib import Path
from uuid import uuid4


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import build_api_runtime  # noqa: E402
from app.api.execution import execute_message  # noqa: E402
from app.asyncio_compat import run_async  # noqa: E402
from app.config import settings  # noqa: E402


SCENARIOS = {
    "class_context": (
        "Create a short play-based activity for the current class. Use the class "
        "context, but do not search EYLF or prior records and do not save anything."
    ),
    "eylf_rag": (
        "Using only EYLF, explain how play-based learning supports children's "
        "agency. Cite the retrieved source and do not save anything."
    ),
    "observation_approval": (
        "Save this complete objective observation as a draft after showing me the "
        "exact fields for approval: On 25 August 2026 at 9:00 am in the outdoor "
        "area, Child 01 stacked four blocks, then invited Child 02 to add another. "
        "The educator supplied more blocks."
    ),
    "draft_save_followup": (
        "Create a short play-based learning activity for preschool children using "
        "natural materials. Do not save it yet.",
        "Save the activity plan you just created as a draft educational record.",
    ),
    "versioned_draft_save": (
        "Create version one of a preschool activity called LEAF-MOSAIC using leaves "
        "and bark. Do not save it.",
        "Create a different version two called WATER-LAB using pouring containers. "
        "Keep both versions and do not save either yet.",
        "Save the first version, LEAF-MOSAIC, as a draft educational record.",
    ),
}


APPROVAL_PREVIEW_EXPECTATIONS = {
    "versioned_draft_save": {
        "required": ("leaf-mosaic",),
        "forbidden": ("water-lab",),
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected live phase-one scenarios. This consumes provider quota."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=["class_context"],
        help="Default is the cheapest single scenario. eylf_rag uses an embedding call.",
    )
    parser.add_argument("--teacher-id", default="teacher-001")
    parser.add_argument("--class-id", default="kangaroo-room")
    return parser.parse_args()


def validate_configuration(args: argparse.Namespace) -> None:
    missing = []
    if not settings.database_url or not settings.checkpoint_database_url:
        missing.append("PostgreSQL URLs")
    if not settings.model_api_key:
        missing.append("MODEL_API_KEY")
    if "eylf_rag" in args.scenarios and not settings.embedding_api_key:
        missing.append("EMBEDDING_API_KEY")
    if missing:
        raise RuntimeError("Missing configuration: " + ", ".join(missing))


async def run(args: argparse.Namespace) -> int:
    runtime = await build_api_runtime()
    try:
        failures = 0
        for scenario_name in args.scenarios:
            # Evaluation scenarios must not inherit conversation history or
            # checkpoints from one another.
            session_id = str(uuid4())
            thread_id = str(uuid4())
            await runtime.store.create_conversation_session(
                session_id=session_id,
                thread_id=thread_id,
                teacher_id=args.teacher_id,
                class_id=args.class_id,
            )
            scenario = SCENARIOS[scenario_name]
            messages = scenario if isinstance(scenario, tuple) else (scenario,)
            request_id = ""
            for turn_number, message in enumerate(messages, start=1):
                request_id = str(uuid4())
                await runtime.store.create_conversation_run(
                    request_id=request_id,
                    session_id=session_id,
                )
                await execute_message(
                    runtime=runtime,
                    request_id=request_id,
                    session_id=session_id,
                    thread_id=thread_id,
                    teacher_id=args.teacher_id,
                    class_id=args.class_id,
                    message=message,
                )
                turn = await runtime.store.get_conversation_run(request_id)
                print(
                    f"\n[{scenario_name} turn={turn_number}] "
                    f"status={turn['status'] if turn else 'missing'}"
                )
            run_record = await runtime.store.get_conversation_run(request_id)
            result = await runtime.store.get_conversation_run_result(request_id)
            status = run_record["status"] if run_record else "missing"
            print(f"\n[{scenario_name}] status={status}")
            print(f"session_id={session_id}")
            if result:
                print(result["draft"]["content"])
                print(f"citations={len(result.get('citations', []))}")
                approval = result.get("approval", {})
                if approval.get("status") == "required":
                    print("approval_required=true")
                    print(f"action_id={approval.get('action_id')}")
                    print("No write was executed; review the draft and approve through the API.")
                    preview_text = str(approval.get("preview", {})).casefold()
                    expectation = APPROVAL_PREVIEW_EXPECTATIONS.get(scenario_name)
                    if expectation:
                        missing = [
                            term for term in expectation["required"] if term not in preview_text
                        ]
                        forbidden = [
                            term for term in expectation["forbidden"] if term in preview_text
                        ]
                        print(
                            "artifact_reference_check="
                            + ("PASS" if not missing and not forbidden else "FAIL")
                        )
                        if missing or forbidden:
                            failures += 1
            expected = (
                {"waiting_for_approval"}
                if scenario_name in {
                    "observation_approval",
                    "draft_save_followup",
                    "versioned_draft_save",
                }
                else {"completed"}
            )
            if status not in expected:
                failures += 1
        return 1 if failures else 0
    finally:
        await runtime.close()


def main() -> int:
    args = parse_args()
    validate_configuration(args)
    return run_async(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
