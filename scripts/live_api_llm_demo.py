"""Run one real-LLM request through FastAPI, LangGraph, SSE, and Draft API."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api import build_api_runtime
from app.config import settings
from app.main import create_app
from app.schemas import RerankerMode, RetrievalMode
from app.services import (
    BM25KnowledgeIndex,
    ChatCompletionsModelProvider,
    KnowledgeRetriever,
    PolicyRAGService,
)
from app.workflows import build_main_graph, build_policy_rag_graph


DEFAULT_QUESTION = "What does the EYLF say about play-based learning?"
CHUNKS_PATH = PROJECT_ROOT / "data/knowledge/processed/chunks.jsonl"


def _check_configuration() -> None:
    missing = []
    if not settings.model_base_url or "example" in settings.model_base_url:
        missing.append("MODEL_BASE_URL")
    if not settings.model_api_key or settings.model_api_key.startswith("replace-"):
        missing.append("MODEL_API_KEY")
    if not settings.model_name:
        missing.append("MODEL_NAME")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Please configure {names} in .env before running this demo.")
    if not CHUNKS_PATH.exists():
        raise RuntimeError(
            "Local knowledge chunks are missing. Run: "
            "python scripts/ingest_knowledge.py --output "
            "data/knowledge/processed/chunks.jsonl"
        )


def _build_live_policy_graph(runtime) -> None:
    """Use local BM25 evidence while keeping routing and answering on the real LLM."""
    lexical_index = BM25KnowledgeIndex.from_jsonl(
        CHUNKS_PATH,
        project_root=PROJECT_ROOT,
    )
    retriever = KnowledgeRetriever(lexical_index=lexical_index)
    policy_service = PolicyRAGService(
        retriever=retriever,
        model_provider=ChatCompletionsModelProvider(),
        top_k=3,
        retrieval_mode=RetrievalMode.BM25,
        reranker=RerankerMode.LEXICAL,
    )
    runtime.graph = build_main_graph(
        policy_workflow=build_policy_rag_graph(service=policy_service),
        checkpointer=runtime.checkpointer,
        long_memory_store=runtime.store,
        learning_record_store=runtime.store,
    )


def _parse_sse(body: str):
    for frame in body.strip().split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        fields = {}
        for line in frame.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                fields[key] = value
        if "data" in fields:
            yield json.loads(fields["data"])


def _print_events(events) -> None:
    print("\n=== SSE events ===")
    for event in events:
        event_name = event["event"]
        data = event.get("data", {})
        if event_name == "trace" and data.get("origin") == "graph":
            print(
                f"[{event['sequence']}] trace | "
                f"{data.get('step')}: {data.get('message')}"
            )
        else:
            print(f"[{event['sequence']}] {event_name} | {data}")


def run_demo(question: str) -> int:
    _check_configuration()
    print(f"model={settings.model_name}")
    print(f"question={question}")
    print("Calling the real model; this may take a little while...")

    with TemporaryDirectory(prefix="eduflow-live-api-") as directory:
        temp_path = Path(directory)
        runtime = build_api_runtime(
            database_path=temp_path / "eduflow.sqlite3",
            checkpoint_database_path=temp_path / "checkpoints.sqlite3",
        )
        _build_live_policy_graph(runtime)
        application = create_app(runtime_factory=lambda: runtime)

        with TestClient(application) as client:
            session_response = client.post("/sessions", json={})
            session_response.raise_for_status()
            session_id = session_response.json()["session_id"]

            message_response = client.post(
                f"/sessions/{session_id}/messages",
                json={"message": question},
            )
            message_response.raise_for_status()
            request_id = message_response.json()["request_id"]

            events_response = client.get(
                f"/sessions/{session_id}/events",
                params={"request_id": request_id},
            )
            events_response.raise_for_status()
            _print_events(list(_parse_sse(events_response.text)))

            draft_response = client.get(
                f"/sessions/{session_id}/drafts/{request_id}"
            )
            if draft_response.status_code != 200:
                print("\n=== No final draft ===")
                print(json.dumps(draft_response.json(), indent=2, ensure_ascii=False))
                return 1

            result = draft_response.json()
            print("\n=== Real LLM answer ===")
            print(result["draft"]["content"])
            print("\n=== Citations ===")
            if not result["citations"]:
                print("(none)")
            for index, citation in enumerate(result["citations"], start=1):
                print(
                    f"[{index}] {citation.get('title') or citation['source']} | "
                    f"section={citation.get('section')} | page={citation.get('page')}"
                )
            print(f"\nstatus={result['status']}")
            print(f"session_id={session_id}")
            print(f"request_id={request_id}")
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a real policy question through the EduFlow FastAPI and "
            "LangGraph stack."
        )
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Policy/EYLF question to ask.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_demo(args.question)
    except Exception as error:
        print("\nLIVE_API_LLM_DEMO_FAILED")
        print(f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
