#!/usr/bin/env python3
"""Live Gemini routing checks for the five extended EasyTeaching capabilities."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4
from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import BoundedWorkerRunner, DEFAULT_WORKER_PROFILES, WorkerRegistry  # noqa: E402
from app.api.checkpoint_config import checkpoint_config  # noqa: E402
from app.api.execution import execute_message  # noqa: E402
from app.api.runtime import build_api_runtime  # noqa: E402
from app.asyncio_compat import run_async  # noqa: E402
from app.schemas import GraphState  # noqa: E402
from app.services import ChatCompletionsModelProvider, ContextManager  # noqa: E402
from app.services.document_reader import UploadedDocumentReader  # noqa: E402
from app.services.file_assets import LocalUploadedFileStore  # noqa: E402
from app.services.scoped_knowledge import ScopedKnowledgeStore  # noqa: E402
from app.services.official_web_search import (  # noqa: E402
    OfficialSearchResult,
    OfficialWebSearchResponse,
)
from app.services.transcription import TranscriptSegment, TranscriptionResult  # noqa: E402
from app.tools.controlled_tools.official_web import (  # noqa: E402
    build_official_web_search_tool,
)
from app.tools.controlled_tools.voice_note import (  # noqa: E402
    build_transcribe_voice_note_tool,
)
from app.tools import build_default_tool_registry  # noqa: E402
from app.workflows import build_main_react_graph  # noqa: E402


class NoOpMemoryExtractor:
    async def decide_async(self, **_):
        return []


class DeterministicOfficialSearch:
    async def search(self, query, *, domains=None, top_k=5):
        del domains, top_k
        return OfficialWebSearchResponse(
            query=query,
            results=[OfficialSearchResult(
                title="ACECQA – National Quality Standard",
                snippet="The National Quality Standard sets a national benchmark for early childhood education and care.",
                url="https://www.acecqa.gov.au/nqf/national-quality-standard",
                domain="www.acecqa.gov.au",
            )],
        )


class DeterministicTranscriber:
    async def transcribe(self, path, *, language=None):
        del path
        text = "A child arranged three leaves by size and invited a peer to add another leaf."
        return TranscriptionResult(
            text=text,
            language=language or "en",
            duration_seconds=4.0,
            segments=[TranscriptSegment(start_seconds=0, end_seconds=4.0, text=text)],
        )


class InMemoryStore:
    def __init__(self):
        self.actions = []

    async def list_profile_memories(self, **_):
        return []

    async def list_memories_for_owners(self, **_):
        return []

    async def get_conversation_workspace(self, **_):
        return {"recent_artifacts": [], "recent_records": []}

    async def get_class_context(self, **_):
        return {
            "centre_id": "centre-live",
            "class_id": "kangaroo-room",
            "name": "Kangaroo Room",
            "age_group": "3-5 years",
            "child_count": 18,
            "current_focus": ["outdoor collaborative play"],
            "children": [],
            "class_memories": [],
        }

    async def query_records(self, **_):
        return [
            {"record_type": "observation", "status": "draft", "observed_at": "2026-08-28T10:00:00"},
            {"record_type": "observation", "status": "final", "observed_at": "2026-08-27T10:00:00"},
            {"record_type": "educational_record", "status": "final", "created_at": "2026-08-26T10:00:00"},
        ]

    async def get_conversation_run_result(self, *_args, **_kwargs):
        return None

    async def create_tool_action_request(self, **values):
        action = {"action_id": str(uuid4()), **values}
        self.actions.append(action)
        return action


async def build_memory_runtime(provider):
    store = InMemoryStore()
    file_store = LocalUploadedFileStore(PROJECT_ROOT / "data" / "local" / "extended-live-uploads")
    scoped = ScopedKnowledgeStore(
        root=PROJECT_ROOT / "data" / "local" / "extended-live-knowledge",
        file_store=file_store,
        document_reader=UploadedDocumentReader(),
    )
    registry = build_default_tool_registry(
        store,
        file_store=file_store,
        document_reader=UploadedDocumentReader(),
        scoped_knowledge=scoped,
    )
    registry.register(build_official_web_search_tool(DeterministicOfficialSearch()))
    registry.register(build_transcribe_voice_note_tool(file_store, DeterministicTranscriber()))
    workers = WorkerRegistry(DEFAULT_WORKER_PROFILES)
    checkpointer = MemorySaver()
    graph = build_main_react_graph(
        model_provider=provider,
        registry=registry,
        worker_registry=workers,
        worker_runner=BoundedWorkerRunner(
            provider=provider,
            tool_registry=registry,
            worker_registry=workers,
        ),
        context_manager=ContextManager(long_term_memory_reader=store),
        checkpointer=checkpointer,
        long_memory_extractor=NoOpMemoryExtractor(),
        long_memory_store=store,
        max_steps=8,
        max_tool_calls=12,
    )
    return SimpleNamespace(
        store=store,
        file_store=file_store,
        tool_registry=registry,
        checkpointer=checkpointer,
        graph=graph,
        close=lambda: asyncio.sleep(0),
    )


SCENARIOS = (
    ("read_document", "Read the uploaded document with file_id {document_file_id} and briefly summarise its supervision rule. Do not add it to the knowledge base or save anything.", "read_uploaded_document", "completed"),
    ("ingest_document", "Add the uploaded document with file_id {document_file_id} to this class's knowledge base with the title 'Live Garden Supervision Policy'. Show me the approval preview first.", "ingest_uploaded_document", "waiting_for_approval"),
    ("analyse_records", "Count this class's authorised learning records by status. Return a short summary and do not display the full record text.", "analyse_learning_records", "completed"),
    ("official_search", "Search current official Australian early-childhood guidance for the National Quality Standard and cite the official page. Do not use Google Drive.", "search_official_web", "completed"),
    ("transcribe_voice", "Transcribe the uploaded voice note with file_id {audio_file_id}. Return the transcript only and do not save an observation.", "transcribe_voice_note", "completed"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run live Gemini checks for five extended Tools.")
    parser.add_argument("--scenario", action="append", choices=[item[0] for item in SCENARIOS])
    parser.add_argument("--memory", action="store_true", help="Use in-memory store/checkpoints while keeping real Gemini.")
    return parser.parse_args()


async def run(args) -> int:
    provider = ChatCompletionsModelProvider()
    runtime = await build_memory_runtime(provider) if args.memory else await build_api_runtime()
    try:
        if runtime.tool_registry.get("search_official_web") is None:
            runtime.tool_registry.register(build_official_web_search_tool(DeterministicOfficialSearch()))
        if runtime.tool_registry.get("transcribe_voice_note") is None:
            runtime.tool_registry.register(
                build_transcribe_voice_note_tool(runtime.file_store, DeterministicTranscriber())
            )
        workers = WorkerRegistry(DEFAULT_WORKER_PROFILES)
        runtime.graph = build_main_react_graph(
            model_provider=provider,
            registry=runtime.tool_registry,
            worker_registry=workers,
            worker_runner=BoundedWorkerRunner(
                provider=provider,
                tool_registry=runtime.tool_registry,
                worker_registry=workers,
            ),
            context_manager=ContextManager(long_term_memory_reader=runtime.store),
            checkpointer=runtime.checkpointer,
            long_memory_extractor=NoOpMemoryExtractor(),
            long_memory_store=runtime.store,
            max_steps=8,
            max_tool_calls=12,
        )
        selected = set(args.scenario or [item[0] for item in SCENARIOS])
        failures = 0
        for name, template, expected_tool, expected_status in SCENARIOS:
            if name not in selected:
                continue
            session_id, thread_id = str(uuid4()), str(uuid4())
            if not args.memory:
                await runtime.store.create_conversation_session(
                    session_id=session_id,
                    thread_id=thread_id,
                    teacher_id="teacher-001",
                    class_id="kangaroo-room",
                )
            document = runtime.file_store.save_bytes(
                filename="garden-supervision.txt",
                content_type="text/plain",
                content=b"Before garden play, educators complete a boundary check and maintain active supervision.",
                teacher_id="teacher-001",
                class_id="kangaroo-room",
                session_id=session_id,
            )
            audio = runtime.file_store.save_bytes(
                filename="observation.wav",
                content_type="audio/wav",
                content=b"RIFF-deterministic-live-routing-fixture",
                teacher_id="teacher-001",
                class_id="kangaroo-room",
                session_id=session_id,
            )
            if name == "analyse_records" and not args.memory:
                await runtime.store.save_observation(
                    teacher_id="teacher-001",
                    class_id="kangaroo-room",
                    child_ids=[],
                    observed_at=datetime(2026, 8, 28, 10, 0),
                    setting="Live evaluation garden",
                    objective_text="A child arranged three leaves from smallest to largest.",
                    educator_actions=None,
                    status="draft",
                    source_request_id=None,
                    idempotency_key=f"extended-live-{uuid4()}",
                )
            request_id = str(uuid4())
            message = template.format(document_file_id=document.file_id, audio_file_id=audio.file_id)
            if args.memory:
                result = await runtime.graph.ainvoke(
                    {
                        "request_id": request_id,
                        "session_id": session_id,
                        "thread_id": thread_id,
                        "teacher_id": "teacher-001",
                        "class_id": "kangaroo-room",
                        "user_message": message,
                    },
                    config=checkpoint_config(thread_id),
                )
                direct_state = GraphState.model_validate(result)
                run_record = {"status": (
                    "waiting_for_approval"
                    if direct_state.workflow_status.value == "waiting_for_approval"
                    else direct_state.workflow_status.value
                )}
            else:
                await runtime.store.create_conversation_run(request_id=request_id, session_id=session_id)
                await execute_message(
                    runtime=runtime,
                    request_id=request_id,
                    session_id=session_id,
                    thread_id=thread_id,
                    teacher_id="teacher-001",
                    class_id="kangaroo-room",
                    message=message,
                )
                run_record = await runtime.store.get_conversation_run(request_id)
            snapshot = await runtime.graph.aget_state(checkpoint_config(thread_id))
            state = GraphState.model_validate(snapshot.values)
            tools = [item.capability_name for item in state.observations.values()]
            if state.approval.tool_name:
                tools.append(state.approval.tool_name)
            status = str(run_record.get("status") if run_record else "missing")
            passed = expected_tool in tools and status == expected_status
            failures += 0 if passed else 1
            print(
                f"[{name}] {'PASS' if passed else 'FAIL'} status={status} "
                f"expected_status={expected_status} tools={tools} expected_tool={expected_tool} "
                f"steps={state.react_step}"
            )
            if state.draft:
                print("  answer=" + " ".join(state.draft.content.split())[:320])
            if state.errors:
                print("  errors=" + "; ".join(f"{item.code}:{item.message}" for item in state.errors))
        return 1 if failures else 0
    finally:
        await provider.client.aclose()
        await runtime.close()


if __name__ == "__main__":
    raise SystemExit(run_async(run(parse_args())))
