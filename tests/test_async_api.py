from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas import Draft, GraphState, RunStatus, WorkflowStatus


class MemoryStore:
    def __init__(self):
        self.sessions = {}
        self.runs = {}
        self.results = {}
        self.events = {}

    async def close(self):
        return None

    async def list_conversation_runs(self, *, statuses=None):
        values = list(self.runs.values())
        return [item for item in values if not statuses or item["status"] in statuses]

    async def create_conversation_session(self, **values):
        record = {**values, "status": "active", "created_at": "now"}
        self.sessions[record["session_id"]] = record
        return record

    async def get_conversation_session(self, session_id):
        return self.sessions.get(session_id)

    async def create_conversation_run(self, *, request_id, session_id):
        existing = self.runs.get(request_id)
        if existing:
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
                and item["status"] in {"accepted", "running"}
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
        items = self.events.setdefault(values["request_id"], [])
        record = {
            "event_id": f"event-{len(items)}",
            "sequence": len(items),
            **values,
        }
        items.append(record)
        return record

    async def list_conversation_events(self, *, request_id, after_sequence=-1):
        return [
            item
            for item in self.events.get(request_id, [])
            if item["sequence"] > after_sequence
        ]


class RecordingGraph:
    def __init__(self):
        self.calls = []
        self.states = {}

    async def ainvoke(self, state, config):
        self.calls.append((state, config))
        if state is None:
            state = self.states[config["configurable"]["thread_id"]]
        result = GraphState.model_validate(state).model_copy(
            update={
                "workflow_status": WorkflowStatus.COMPLETED,
                "draft": Draft(title="Async draft", content="Completed asynchronously."),
            }
        )
        self.states[result.thread_id] = result
        return result

    async def aget_state(self, config):
        state = self.states.get(config["configurable"]["thread_id"])
        return SimpleNamespace(
            values={} if state is None else state.model_dump(mode="json"),
            next=(),
        )


class FakeRuntime:
    def __init__(self):
        self.store = MemoryStore()
        self.graph = RecordingGraph()
        self.is_closed = False

    async def close(self):
        self.is_closed = True


def test_async_message_lifecycle_idempotency_draft_and_events():
    runtime = FakeRuntime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        session = client.post(
            "/sessions",
            json={"teacher_id": "teacher-1", "class_id": "kangaroo-room"},
        ).json()
        payload = {"message": "Create a plan.", "request_id": "request-1"}
        first = client.post(f"/sessions/{session['session_id']}/messages", json=payload)
        second = client.post(f"/sessions/{session['session_id']}/messages", json=payload)
        draft = client.get(
            f"/sessions/{session['session_id']}/drafts/request-1"
        )
        events = client.get(
            f"/sessions/{session['session_id']}/events",
            params={"request_id": "request-1"},
        )

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["status"] == RunStatus.COMPLETED.value
    assert len(runtime.graph.calls) == 1
    assert draft.status_code == 200
    assert draft.json()["draft"]["is_draft"] is True
    assert events.status_code == 200
    assert "event: completed" in events.text
    assert runtime.is_closed is True


def test_same_session_busy_error_rejects_a_second_active_run():
    runtime = FakeRuntime()
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        runtime.store.runs["active"] = {
            "request_id": "active",
            "session_id": session_id,
            "status": "running",
        }
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Another task.", "request_id": "request-2"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_busy"
