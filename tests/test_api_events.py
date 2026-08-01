import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langgraph.types import Command

from app.api import build_api_runtime
from app.main import create_app
from app.schemas import (
    Approval,
    ApprovalStatus,
    Draft,
    GraphState,
    RiskLevel,
    TraceEvent,
    WorkflowStatus,
)


class EventGraph:
    def __init__(self) -> None:
        self.paused_state = None

    def invoke(self, state, config):
        if isinstance(state, Command):
            self.paused_state = self.paused_state.model_copy(
                update={
                    "workflow_status": WorkflowStatus.COMPLETED,
                    "approval": self.paused_state.approval.model_copy(
                        update={"status": ApprovalStatus.APPROVED}
                    ),
                }
            )
            return self.paused_state

        self.paused_state = GraphState.model_validate(state).model_copy(
            update={
                "workflow_status": WorkflowStatus.WAITING_FOR_APPROVAL,
                "draft": Draft(title="Outdoor plan", content="Draft content"),
                "approval": Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason="Teacher review is required.",
                ),
            }
        )
        return self.paused_state


class CompleteTraceGraph:
    def invoke(self, state, config):
        return GraphState.model_validate(state).model_copy(
            update={
                "workflow_status": WorkflowStatus.COMPLETED,
                "trace": [
                    TraceEvent(
                        step="initialize",
                        message="Initialized graph state.",
                        metadata={
                            "thread_id": state.thread_id,
                            "unsafe_user_text": "private observation",
                        },
                    ),
                    TraceEvent(
                        step="intent_router",
                        message="Intent routing completed.",
                        metadata={
                            "intent": "activity_planning",
                            "confidence": 0.95,
                            "code": "rate_limited",
                            "attempts": 3,
                            "max_attempts": 3,
                            "retry_exhausted": True,
                            "status_code": 429,
                            "body": "private provider response",
                            "reason": "Contains a private child name.",
                            "clarification_question": "Is this about Child A?",
                        },
                    ),
                    TraceEvent(
                        step="planning_react",
                        message="Planning workflow completed.",
                        metadata={
                            "stop_reason": "completed",
                            "observations": [
                                {
                                    "tool_name": "get_class_profile",
                                    "success": True,
                                    "error_code": None,
                                    "tool_output": "private profile data",
                                }
                            ],
                        },
                    ),
                ],
            }
        )


class TraceResumeGraph:
    def __init__(self) -> None:
        self.paused_state = None

    def invoke(self, state, config):
        if isinstance(state, Command):
            self.paused_state = self.paused_state.model_copy(
                update={
                    "workflow_status": WorkflowStatus.COMPLETED,
                    "approval": self.paused_state.approval.model_copy(
                        update={"status": ApprovalStatus.APPROVED}
                    ),
                    "trace": [
                        *self.paused_state.trace,
                        TraceEvent(
                            step="approval_gate",
                            message="Teacher approved draft.",
                            metadata={"decision": "approve"},
                        ),
                    ],
                }
            )
            return self.paused_state

        self.paused_state = GraphState.model_validate(state).model_copy(
            update={
                "workflow_status": WorkflowStatus.WAITING_FOR_APPROVAL,
                "draft": Draft(title="Draft", content="Draft content"),
                "approval": Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                ),
                "trace": [
                    TraceEvent(step="initialize", message="Initialized."),
                    TraceEvent(
                        step="documentation_draft",
                        message="Generated de-identified draft.",
                    ),
                ],
            }
        )
        return self.paused_state


class FailingTraceGraph:
    def __init__(self) -> None:
        self.checkpoint_state = None

    def invoke(self, state, config):
        self.checkpoint_state = GraphState.model_validate(state).model_copy(
            update={
                "workflow_status": WorkflowStatus.ROUTED,
                "trace": [
                    TraceEvent(step="initialize", message="Initialized."),
                    TraceEvent(
                        step="intent_router",
                        message="Intent routing completed.",
                        metadata={"intent": "activity_planning"},
                    ),
                ],
            }
        )
        raise RuntimeError("private provider failure detail")

    def get_state(self, config):
        return SimpleNamespace(values=self.checkpoint_state.model_dump())


def _client(tmp_path):
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    runtime.graph = EventGraph()
    application = create_app(runtime_factory=lambda: runtime)
    return TestClient(application)


def _client_with_graph(tmp_path, graph):
    runtime = build_api_runtime(
        database_path=tmp_path / "eduflow.sqlite3",
        checkpoint_database_path=tmp_path / "checkpoints.sqlite3",
    )
    runtime.graph = graph
    application = create_app(runtime_factory=lambda: runtime)
    return runtime, TestClient(application)


def _events(response):
    frames = [frame for frame in response.text.strip().split("\n\n") if frame]
    parsed = []
    for frame in frames:
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        parsed.append(
            {
                "id": lines["id"],
                "event": lines["event"],
                "data": json.loads(lines["data"]),
            }
        )
    return parsed


def test_sse_replays_ordered_events_until_approval_pause(tmp_path) -> None:
    client = _client(tmp_path)

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Write a plan.", "request_id": "req-events"},
        )
        response = client.get(
            f"/sessions/{session_id}/events",
            params={"request_id": "req-events"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response)
        assert [event["event"] for event in events] == [
            "run_started",
            "draft_ready",
            "approval_required",
        ]
        assert [event["data"]["sequence"] for event in events] == [0, 1, 2]
        assert all(event["id"] == event["data"]["event_id"] for event in events)
        assert events[-1]["data"]["data"]["risk_level"] == "L2_controlled_write"


def test_sse_can_resume_after_the_last_received_sequence(tmp_path) -> None:
    client = _client(tmp_path)

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Write a plan.", "request_id": "req-resume"},
        )
        response = client.get(
            f"/sessions/{session_id}/events",
            params={"request_id": "req-resume", "after_sequence": 0},
        )

        events = _events(response)
        assert [event["data"]["sequence"] for event in events] == [1, 2]


def test_sse_continues_with_approval_and_completion_events(tmp_path) -> None:
    client = _client(tmp_path)

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Write a plan.", "request_id": "req-approved-events"},
        )
        client.post(
            f"/sessions/{session_id}/approvals",
            json={"request_id": "req-approved-events", "decision": "approve"},
        )
        response = client.get(
            f"/sessions/{session_id}/events",
            params={"request_id": "req-approved-events", "after_sequence": 2},
        )

        events = _events(response)
        assert [event["event"] for event in events] == [
            "trace",
            "draft_ready",
            "completed",
        ]
        assert [event["data"]["sequence"] for event in events] == [3, 4, 5]


def test_sse_does_not_expose_another_sessions_request(tmp_path) -> None:
    client = _client(tmp_path)

    with client:
        first = client.post("/sessions", json={}).json()["session_id"]
        second = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{first}/messages",
            json={"message": "Write a plan.", "request_id": "req-private-events"},
        )
        response = client.get(
            f"/sessions/{second}/events",
            params={"request_id": "req-private-events"},
        )

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "request_not_found"


def test_sse_persists_every_graph_trace_with_safe_metadata(tmp_path) -> None:
    runtime, client = _client_with_graph(tmp_path, CompleteTraceGraph())

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Plan an activity.", "request_id": "req-full-trace"},
        )
        response = client.get(
            f"/sessions/{session_id}/events",
            params={"request_id": "req-full-trace"},
        )

        events = _events(response)
        trace_events = [event["data"]["data"] for event in events if event["event"] == "trace"]
        assert [event["trace_index"] for event in trace_events] == [0, 1, 2]
        assert [event["step"] for event in trace_events] == [
            "initialize",
            "intent_router",
            "planning_react",
        ]
        assert trace_events[0]["metadata"] == {
            "thread_id": trace_events[0]["metadata"]["thread_id"]
        }
        assert trace_events[1]["metadata"] == {
            "intent": "activity_planning",
            "confidence": 0.95,
            "code": "rate_limited",
            "attempts": 3,
            "max_attempts": 3,
            "retry_exhausted": True,
            "status_code": 429,
        }
        assert trace_events[2]["metadata"]["observations"] == [
            {
                "tool_name": "get_class_profile",
                "success": True,
                "error_code": None,
            }
        ]

        stored = runtime.store.list_conversation_events(request_id="req-full-trace")
        assert len([event for event in stored if event["event"] == "trace"]) == 3


def test_approval_resume_only_persists_new_graph_trace_entries(tmp_path) -> None:
    runtime, client = _client_with_graph(tmp_path, TraceResumeGraph())

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Write a record.", "request_id": "req-trace-resume"},
        )
        client.post(
            f"/sessions/{session_id}/approvals",
            json={"request_id": "req-trace-resume", "decision": "approve"},
        )

        stored = runtime.store.list_conversation_events(request_id="req-trace-resume")
        graph_traces = [
            event["data"]
            for event in stored
            if event["event"] == "trace"
            and event["data"].get("origin") == "graph"
        ]
        assert [event["trace_index"] for event in graph_traces] == [0, 1, 2]
        assert [event["step"] for event in graph_traces] == [
            "initialize",
            "documentation_draft",
            "approval_gate",
        ]


def test_failed_run_recovers_partial_trace_from_checkpoint(tmp_path) -> None:
    _, client = _client_with_graph(tmp_path, FailingTraceGraph())

    with client:
        session_id = client.post("/sessions", json={}).json()["session_id"]
        client.post(
            f"/sessions/{session_id}/messages",
            json={"message": "Plan an activity.", "request_id": "req-failed-trace"},
        )
        response = client.get(
            f"/sessions/{session_id}/events",
            params={"request_id": "req-failed-trace"},
        )

        events = _events(response)
        assert [event["event"] for event in events] == [
            "run_started",
            "trace",
            "trace",
            "failed",
        ]
        assert [
            event["data"]["data"]["step"]
            for event in events
            if event["event"] == "trace"
        ] == ["initialize", "intent_router"]
        assert "private provider failure detail" not in response.text
