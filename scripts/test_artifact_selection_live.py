#!/usr/bin/env python3
"""Live LLM check for selecting among synthetic conversation artefact versions."""

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import MainReActAgent  # noqa: E402
from app.services import ChatCompletionsModelProvider  # noqa: E402
from app.tools.controlled_tools.records import (  # noqa: E402
    build_save_educational_record_tool,
)


WORKSPACE = """Conversation workspace references (server-owned identifiers):
- Artifact: number=1; relation=previous; source_request_id=synthetic-leaf-v1; title=LEAF-MOSAIC; content_chars=900
- Artifact: number=2; relation=latest/current; source_request_id=synthetic-water-v2; title=WATER-LAB; content_chars=850
"""

CASES = {
    "first": ("Save the first version as a draft educational record.", "synthetic-leaf-v1"),
    "previous": ("Save the previous version, not the current one.", "synthetic-leaf-v1"),
    "latest": ("Save the latest version as a draft educational record.", "synthetic-water-v2"),
    "ambiguous": (
        "Save one of these two versions as a draft educational record, but I have "
        "not said which version I want.",
        None,
    ),
}


async def main() -> int:
    provider = ChatCompletionsModelProvider()
    agent = MainReActAgent(provider)
    tool = build_save_educational_record_tool(object())
    failures = 0
    try:
        for name, (message, expected_id) in CASES.items():
            decision = await agent.decide(
                user_message=message,
                conversation_context=WORKSPACE,
                observations={},
                available_tools=[tool],
                available_workers=[],
                current_step=0,
                max_steps=4,
            )
            actual_id = None
            if decision.tool_calls:
                actual_id = decision.tool_calls[0].arguments.get("source_request_id")
            passed = (
                actual_id == expected_id
                if expected_id is not None
                else bool(decision.clarification_question) and not decision.tool_calls
            )
            failures += int(not passed)
            print(
                f"[{name}] {'PASS' if passed else 'FAIL'} "
                f"expected={expected_id or 'clarification'} "
                f"actual={actual_id or ('clarification' if decision.clarification_question else 'other')}"
            )
    finally:
        await provider.client.aclose()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
