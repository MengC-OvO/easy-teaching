"""Implementation behind the synthetic-only end-to-end privacy demo.

The Safety Gateway uses the real Qwen adapter. The ReAct graph is replaced by a
deterministic local node so this smoke test needs neither PostgreSQL nor an
external model-provider credential.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.integrations.privacy_gateway_client import PrivacyGatewayClient
from app.main import create_app
from app.schemas import Draft, GraphState, WorkflowStatus


SYNTHETIC_INPUT = (
    "Synthetic child Aria Example will be absent tomorrow. "
    "Contact 0491 570 006 or aria@example.test."
)


class MemoryStore:
    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self.runs: dict[str, dict] = {}
        self.results: dict[str, dict] = {}
        self.events: dict[str, list[dict]] = {}

    async def close(self) -> None:
        return None

    async def list_conversation_runs(self, *, statuses=None):
        values = list(self.runs.values())
        return [item for item in values if not statuses or item["status"] in statuses]

    async def create_conversation_session(self, **values):
        record = {**values, "status": "active", "created_at": "synthetic-smoke"}
        self.sessions[record["session_id"]] = record
        return record

    async def get_conversation_session(self, session_id):
        return self.sessions.get(session_id)

    async def create_conversation_run(self, *, request_id, session_id):
        existing = self.runs.get(request_id)
        if existing is not None:
            return {**existing, "created": False}
        record = {
            "request_id": request_id,
            "session_id": session_id,
            "status": "accepted",
            "created": True,
        }
        self.runs[request_id] = record
        return record

    async def get_conversation_run(self, request_id):
        record = self.runs.get(request_id)
        return None if record is None else {**record, "created": False}

    async def get_active_conversation_run(self, session_id):
        return next(
            (
                item
                for item in self.runs.values()
                if item["session_id"] == session_id
                and item["status"] in {"accepted", "running", "waiting_for_approval"}
            ),
            None,
        )

    async def update_conversation_run_status(self, request_id, status):
        self.runs[request_id]["status"] = status
        return self.runs[request_id]

    async def save_conversation_run_result(self, **values):
        self.results[values["request_id"]] = values
        return values

    async def get_conversation_run_result(self, request_id):
        return self.results.get(request_id)

    async def append_conversation_event(self, **values):
        records = self.events.setdefault(values["request_id"], [])
        record = {
            "event_id": f"synthetic-event-{len(records)}",
            "sequence": len(records),
            **values,
        }
        records.append(record)
        return record

    async def list_conversation_events(self, *, request_id, after_sequence=-1):
        return [
            item
            for item in self.events.get(request_id, [])
            if item["sequence"] > after_sequence
        ]


class DeterministicDemoGraph:
    """Expose the redacted Agent boundary without calling an external LLM."""

    def __init__(self) -> None:
        self.submitted_state: dict | None = None
        self.completed_state: GraphState | None = None

    async def ainvoke(self, state, config):
        del config
        self.submitted_state = dict(state)
        current = GraphState.model_validate(state)
        self.completed_state = current.model_copy(
            update={
                "workflow_status": WorkflowStatus.COMPLETED,
                "draft": Draft(
                    title="Synthetic absence-note draft",
                    content=f"Teacher-facing draft: {current.user_message}",
                ),
            }
        )
        return self.completed_state

    async def aget_state(self, config):
        del config
        values = (
            {}
            if self.completed_state is None
            else self.completed_state.model_dump(mode="json")
        )
        return SimpleNamespace(values=values, next=())


class RecordingGatewayClient(PrivacyGatewayClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.inspection = None
        self.restore_request = None
        self.restore_response = None

    async def inspect(self, request):
        self.inspection = await super().inspect(request)
        return self.inspection

    async def restore(self, request):
        self.restore_request = request
        self.restore_response = await super().restore(request)
        return self.restore_response


class DemoRuntime:
    def __init__(self, gateway_url: str) -> None:
        self.store = MemoryStore()
        self.graph = DeterministicDemoGraph()
        self.privacy_gateway_mode = "enforce"
        self.privacy_gateway_client = RecordingGatewayClient(
            base_url=gateway_url,
            timeout_seconds=30,
        )
        self.is_closed = False

    async def close(self) -> None:
        await self.privacy_gateway_client.aclose()
        await self.store.close()
        self.is_closed = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8010")
    args = parser.parse_args()

    runtime = DemoRuntime(args.gateway_url)
    app = create_app(runtime_factory=lambda: runtime)

    print("\n[1/6] Synthetic input (never production data)")
    print(f"      ORIGINAL : {SYNTHETIC_INPUT}")
    print("\n[2/6] Sending it through the real EasyTeaching FastAPI message route...")

    with TestClient(app) as client:
        session = client.post("/sessions", json={})
        session.raise_for_status()
        session_id = session.json()["session_id"]
        accepted = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": SYNTHETIC_INPUT, "request_id": "synthetic-e2e-001"},
        )
        accepted.raise_for_status()

        inspection = runtime.privacy_gateway_client.inspection
        if inspection is None or runtime.graph.submitted_state is None:
            raise RuntimeError("The real gateway did not produce an inspection")

        print("\n[3/6] Real regex + Qwen v11 safety result")
        print(f"      ACTION   : {inspection.action.value}")
        print(f"      SIGNALS  : {inspection.signals.model_dump(mode='json')}")
        print(f"      ENTITIES : {inspection.entity_counts}")
        print(f"      REDACTED : {runtime.graph.submitted_state['user_message']}")
        print(f"      MAP ID   : {inspection.mapping_id[:12]}... (opaque only)")

        graph_draft = runtime.graph.completed_state
        if graph_draft is None or graph_draft.draft is None:
            raise RuntimeError("The deterministic demo graph produced no draft")
        print("\n[4/6] Text visible inside GraphState / the ReAct boundary")
        print(f"      AGENT IN : {runtime.graph.submitted_state['user_message']}")
        print(f"      AGENT OUT: {graph_draft.draft.content}")
        if "Aria Example" in graph_draft.draft.content:
            raise RuntimeError("Plaintext name leaked into GraphState")

        print("\n[5/6] EasyTeaching called the real /v1/restore endpoint")
        restore_request = runtime.privacy_gateway_client.restore_request
        if restore_request is None:
            raise RuntimeError("Final deterministic restoration was not called")
        print(f"      BEFORE   : {restore_request.text}")

        draft_response = client.get(
            f"/sessions/{session_id}/drafts/synthetic-e2e-001"
        )
        draft_response.raise_for_status()
        final_content = draft_response.json()["draft"]["content"]
        print("\n[6/6] Final teacher-facing FastAPI result")
        print(f"      FINAL    : {final_content}")
        print(f"      RUN      : {runtime.store.runs['synthetic-e2e-001']['status']}")

        if "Aria Example" not in final_content:
            raise RuntimeError("The synthetic name was not restored")
        if "0491 570 006" not in final_content:
            raise RuntimeError("The synthetic reserved phone was not restored")
        if "aria@example.test" not in final_content:
            raise RuntimeError("The synthetic email was not restored")

    print("\nPASS: real Qwen inspection + FastAPI run + redacted GraphState + deterministic restore all succeeded.")


if __name__ == "__main__":
    main()
