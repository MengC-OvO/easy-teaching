import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.main import create_app
from app.api.execution import _recovered_run_status
from app.schemas import Draft, GraphState, RiskLevel, RunStatus, WorkflowStatus
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolRegistry,
    ToolResult,
)
from app.integrations.privacy_gateway_client import PrivacyGatewayUnavailableError
from safety_gateway.contracts import (
    GatewayAction,
    InspectResponse,
    RestoreResponse,
    SafetySignals,
)


class MemoryStore:
    def __init__(self):
        self.sessions = {}
        self.runs = {}
        self.results = {}
        self.events = {}
        self.actions = {}

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

    async def create_conversation_run(self, *, request_id, session_id, task_payload=None):
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

    async def get_tool_action_request(self, action_id):
        return self.actions.get(action_id)

    async def claim_tool_action_request(self, action_id):
        action = self.actions[action_id]
        if action["status"] != "pending":
            raise ValueError("Tool action request has already been decided")
        action["status"] = "executing"
        return dict(action)

    async def finish_tool_action_request(self, action_id, *, status, result=None):
        action = self.actions[action_id]
        action["status"] = status
        action["result"] = result
        return dict(action)


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
        self.privacy_gateway_mode = "disabled"
        self.privacy_gateway_client = None
        self.tool_registry = ToolRegistry()
        self.is_closed = False

    async def close(self):
        self.is_closed = True


class StubRateLimiter:
    def __init__(self, *, allowed=True, error=None):
        self.allowed = allowed
        self.error = error

    async def check(self, identity):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            allowed=self.allowed,
            remaining=0,
            retry_after_seconds=17,
        )


class FakeGatewayClient:
    def __init__(self, result=None, error=None, restore_error=None):
        self.result = result
        self.error = error
        self.restore_error = restore_error
        self.requests = []
        self.restore_requests = []
        self.discarded_mapping_ids = []
    async def inspect(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result

    async def discard(self, mapping_id):
        self.discarded_mapping_ids.append(mapping_id)
        return None

    async def restore(self, request):
        self.restore_requests.append(request)
        if self.restore_error:
            raise self.restore_error
        return RestoreResponse(
            restored_text=request.text.replace("<PERSON_NAME_1>", "Maya Example")
        )


def test_late_graph_failure_preserves_a_durable_completed_checkpoint() -> None:
    completed = GraphState(
        request_id="request-completed-checkpoint",
        session_id="session-1",
        user_message="Create a draft.",
        workflow_status=WorkflowStatus.COMPLETED,
        draft=Draft(title="Recovered", content="Durable teacher-facing result."),
    )
    incomplete = completed.model_copy(
        update={"workflow_status": WorkflowStatus.DRAFTING, "draft": None}
    )

    assert _recovered_run_status(completed) is RunStatus.COMPLETED
    assert _recovered_run_status(incomplete) is RunStatus.FAILED
    assert _recovered_run_status(None) is RunStatus.FAILED


class PlaceholderGraph(RecordingGraph):
    async def ainvoke(self, state, config):
        self.calls.append((state, config))
        result = GraphState.model_validate(state).model_copy(
            update={
                "workflow_status": WorkflowStatus.COMPLETED,
                "draft": Draft(
                    title="EasyTeaching draft",
                    content="A learning plan for <PERSON_NAME_1>.",
                ),
            }
        )
        self.states[result.thread_id] = result
        return result


class FailingGraph(RecordingGraph):
    async def ainvoke(self, state, config):
        self.calls.append((state, config))
        raise RuntimeError("synthetic graph failure")


class ApprovedWriteInput(BaseModel):
    text: str


class ApprovedWriteOutput(BaseModel):
    saved: str


def register_approved_write(runtime: FakeRuntime) -> None:
    runtime.tool_registry.register(
        ToolDefinition(
            name="save_synthetic",
            description="Synthetic approval-gated write.",
            category=ToolCategory.DRAFT,
            input_model=ApprovedWriteInput,
            output_model=ApprovedWriteOutput,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
            permission=ToolPermission.REQUIRE_APPROVAL,
            domain=ToolDomain.LOCAL,
            handler=lambda args: ToolResult.ok(
                data={"saved": args.text},
                risk_level=RiskLevel.L2_CONTROLLED_WRITE,
            ),
        )
    )


def gateway_result(*, action="allow"):
    return InspectResponse(
        request_id=uuid4(),
        action=GatewayAction(action),
        reason_code=f"synthetic_{action}",
        signals=SafetySignals(
            injection_risk="normal" if action == "allow" else "block",
            education_scope="in_scope",
            professional_risk="none",
        ),
        redacted_text="Create a plan for <PERSON_NAME_1>." if action == "allow" else None,
        mapping_id="opaque-mapping-id-123456789" if action == "allow" else None,
        entity_counts={"PERSON_NAME": 1} if action == "allow" else {},
    )


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
    assert runtime.graph.calls[0][0]["privacy_mapping_id"] is None
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


def test_distributed_rate_limit_returns_429_with_retry_after() -> None:
    runtime = FakeRuntime()
    runtime.redis_rate_limiter = StubRateLimiter(allowed=False)
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Create a plan.", "request_id": "limited-1"},
        )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "17"
    assert response.json()["error"]["code"] == "rate_limit_exceeded"
    assert runtime.store.runs == {}
    assert runtime.graph.calls == []


def test_rate_limiter_outage_fails_closed_before_expensive_work() -> None:
    runtime = FakeRuntime()
    runtime.redis_rate_limiter = StubRateLimiter(error=ConnectionError("offline"))
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Create a plan.", "request_id": "limit-down-1"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "rate_limiter_unavailable"
    assert runtime.store.runs == {}
    assert runtime.graph.calls == []


def test_enforce_mode_check_happens_before_run_persistence_and_forwards_only_redacted_text():
    runtime = FakeRuntime()
    runtime.graph = PlaceholderGraph()
    runtime.privacy_gateway_mode = "enforce"
    gateway = FakeGatewayClient(gateway_result())
    runtime.privacy_gateway_client = gateway
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Synthetic child Maya Example needs a plan.", "request_id": "safe-1"},
        )
    assert response.status_code == 202
    submitted_state = runtime.graph.calls[0][0]
    assert submitted_state["user_message"] == "Create a plan for <PERSON_NAME_1>."
    assert submitted_state["privacy_mapping_id"] == "opaque-mapping-id-123456789"
    assert "Maya Example" not in str(submitted_state)
    assert runtime.graph.states[submitted_state["thread_id"]].draft.content == (
        "A learning plan for <PERSON_NAME_1>."
    )
    assert runtime.store.results["safe-1"]["draft"]["content"] == (
        "A learning plan for Maya Example."
    )
    assert len(gateway.restore_requests) == 1


def test_enforce_block_creates_no_run_and_never_invokes_graph():
    runtime = FakeRuntime()
    runtime.privacy_gateway_mode = "enforce"
    runtime.privacy_gateway_client = FakeGatewayClient(gateway_result(action="block"))
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Synthetic direct injection", "request_id": "blocked-1"},
        )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "safety_blocked"
    assert runtime.store.runs == {}
    assert runtime.graph.calls == []


def test_enforce_gateway_failure_creates_no_run_and_fails_closed():
    runtime = FakeRuntime()
    runtime.privacy_gateway_mode = "enforce"
    runtime.privacy_gateway_client = FakeGatewayClient(
        error=PrivacyGatewayUnavailableError("synthetic unavailable")
    )
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Synthetic family request", "request_id": "failed-1"},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "privacy_gateway_unavailable"
    assert runtime.store.runs == {}
    assert runtime.graph.calls == []


def test_enforce_graph_failure_discards_mapping_and_exposes_no_draft():
    runtime = FakeRuntime()
    runtime.graph = FailingGraph()
    runtime.privacy_gateway_mode = "enforce"
    gateway = FakeGatewayClient(gateway_result())
    runtime.privacy_gateway_client = gateway
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Synthetic child Maya Example.", "request_id": "failed-run-1"},
        )

    assert response.status_code == 202
    assert runtime.store.runs["failed-run-1"]["status"] == RunStatus.FAILED.value
    assert runtime.store.results == {}
    assert gateway.discarded_mapping_ids == ["opaque-mapping-id-123456789"]


def test_enforce_restore_failure_marks_run_failed_and_publishes_no_draft():
    runtime = FakeRuntime()
    runtime.graph = PlaceholderGraph()
    runtime.privacy_gateway_mode = "enforce"
    gateway = FakeGatewayClient(
        gateway_result(),
        restore_error=PrivacyGatewayUnavailableError("synthetic restore failure"),
    )
    runtime.privacy_gateway_client = gateway
    app = create_app(runtime_factory=lambda: runtime)
    with TestClient(app) as client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Synthetic child Maya Example.", "request_id": "restore-fail-1"},
        )

    assert response.status_code == 202
    assert runtime.store.runs["restore-fail-1"]["status"] == RunStatus.FAILED.value
    assert runtime.store.results == {}
    event_types = [item["event"] for item in runtime.store.events["restore-fail-1"]]
    assert "draft_ready" not in event_types
    assert "failed" in event_types
    assert gateway.discarded_mapping_ids == ["opaque-mapping-id-123456789"]


def test_approval_executes_only_the_frozen_action_and_completes_run() -> None:
    runtime = FakeRuntime()
    register_approved_write(runtime)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        session_id = client.post(
            "/sessions",
            json={"teacher_id": "teacher-1", "class_id": "kangaroo-room"},
        ).json()["session_id"]
        arguments = {"text": "teacher-reviewed value"}
        serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        runtime.store.runs["approval-request"] = {
            "request_id": "approval-request",
            "session_id": session_id,
            "status": RunStatus.WAITING_FOR_APPROVAL.value,
        }
        runtime.store.actions["action-1"] = {
            "action_id": "action-1",
            "request_id": "approval-request",
            "session_id": session_id,
            "teacher_id": "teacher-1",
            "class_id": "kangaroo-room",
            "tool_name": "save_synthetic",
            "arguments": arguments,
            "arguments_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "status": "pending",
        }
        runtime.store.results["approval-request"] = {
            "request_id": "approval-request",
            "session_id": session_id,
            "draft": {"title": "Confirm", "content": "preview", "is_draft": True},
            "approval": {
                "status": "required",
                "risk_level": "L2_controlled_write",
                "action_id": "action-1",
                "tool_name": "save_synthetic",
                "preview": arguments,
            },
            "citations": [],
        }

        response = client.post(
            f"/sessions/{session_id}/approvals",
            json={"request_id": "approval-request", "decision": "approve"},
        )

    assert response.status_code == 200
    assert runtime.store.actions["action-1"]["status"] == "executed"
    assert runtime.store.results["approval-request"]["approval"]["result"] == {
        "saved": "teacher-reviewed value"
    }
    assert runtime.store.runs["approval-request"]["status"] == RunStatus.COMPLETED.value
    assert runtime.store.events["approval-request"][-1]["event"] == "completed"
