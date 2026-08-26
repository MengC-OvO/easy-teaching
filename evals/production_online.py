"""Production-path online evaluation using synthetic data only.

This suite exercises the real FastAPI lifespan, PostgreSQL store/checkpointer,
Main ReAct model, local RAG indexes and approval execution. It deliberately avoids
real child/family data and does not claim to certify auth or the local privacy model.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.api.runtime import ApiRuntime
from app.main import create_app
from app.services import AsyncEasyTeachingStore, ContextManager
from app.tools import build_default_tool_registry
from app.workflows import build_main_react_graph, build_postgres_checkpointer


class ProductionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    expected: Any = None
    actual: Any = None


class ProductionCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    passed: bool
    latency_ms: float
    status: str
    tools: List[str] = Field(default_factory=list)
    citation_count: int = 0
    answer_preview: str = ""
    checks: List[ProductionCheck] = Field(default_factory=list)
    tool_schema_chars_per_turn: List[int] = Field(default_factory=list)
    observation_view_chars_per_turn: List[int] = Field(default_factory=list)
    conversation_context_chars_per_turn: List[int] = Field(default_factory=list)


class ProductionOnlineSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    passed: int
    pass_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    approval_safety_rate: float
    idempotency_passed: bool
    concurrency_passed: bool
    unnecessary_tool_call_rate: float
    average_tool_schema_chars_per_turn: float
    average_observation_view_chars_per_turn: float
    average_conversation_context_chars_per_turn: float
    slo_passed: bool


class ProductionOnlineReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite: str = "easyteaching-production-online-v1"
    started_at: str
    finished_at: str
    environment: Dict[str, str]
    thresholds: Dict[str, float]
    summary: ProductionOnlineSummary
    results: List[ProductionCaseResult]


def _check(name: str, expected: Any, actual: Any, passed: Optional[bool] = None):
    return ProductionCheck(
        name=name,
        expected=expected,
        actual=actual,
        passed=(expected == actual if passed is None else passed),
    )


def parse_sse_events(text: str) -> List[Dict[str, Any]]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            payload = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _trace_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    tools: List[str] = []
    schema_chars: List[int] = []
    observation_chars: List[int] = []
    context_chars: List[int] = []
    for event in events:
        data = event.get("data") or {}
        if data.get("origin") != "graph":
            continue
        metadata = data.get("metadata") or {}
        if data.get("step") == "merge_observations":
            for item in metadata.get("observations", []):
                name = item.get("tool_name")
                if name and name != "decision_validator":
                    tools.append(str(name))
        if data.get("step") == "main_react":
            if isinstance(metadata.get("tool_schema_chars"), int):
                schema_chars.append(metadata["tool_schema_chars"])
            if isinstance(metadata.get("observation_view_chars"), int):
                observation_chars.append(metadata["observation_view_chars"])
            if isinstance(metadata.get("conversation_context_chars"), int):
                context_chars.append(metadata["conversation_context_chars"])
    return {
        "tools": tools,
        "tool_schema_chars_per_turn": schema_chars,
        "observation_view_chars_per_turn": observation_chars,
        "conversation_context_chars_per_turn": context_chars,
    }


class ProductionOnlineRunner:
    def __init__(
        self,
        *,
        p95_slo_ms: float = 30_000,
        concurrency: int = 3,
        case_ids: Optional[List[str]] = None,
    ):
        self.p95_slo_ms = p95_slo_ms
        self.concurrency = max(1, concurrency)
        self.results: List[ProductionCaseResult] = []
        self.marker = f"PROD-EVAL-{uuid4().hex[:10].upper()}"
        self.case_ids = set(case_ids or [])

    def run(self) -> ProductionOnlineReport:
        started_at = datetime.now(timezone.utc)
        app = create_app(runtime_factory=self._runtime_factory)
        with TestClient(app) as client:
            steps = (
                ("health-validation", lambda: self._health_and_validation(client)),
                ("activity", lambda: self._activity(client)),
                ("policy-rag", lambda: self._policy(client)),
                ("observation-write-query", lambda: self._observation_write_and_query(client)),
                ("versioned-reference", lambda: self._versioned_reference(client)),
                ("cross-framework", lambda: self._cross_framework(client)),
            )
            for label, step in steps:
                if self.case_ids and label not in self.case_ids:
                    continue
                print(f"production_case={label} status=starting", flush=True)
                step()
                print(f"production_case={label} status=finished", flush=True)
            idempotency_passed = True
            if not self.case_ids or "idempotency" in self.case_ids:
                print("production_case=idempotency status=starting", flush=True)
                idempotency_passed = self._idempotency(client)
                print("production_case=idempotency status=finished", flush=True)
            concurrency_passed = True
            if not self.case_ids or "concurrency" in self.case_ids:
                print("production_case=concurrency status=starting", flush=True)
                concurrency_passed = self._concurrency(client)
                print("production_case=concurrency status=finished", flush=True)

        finished_at = datetime.now(timezone.utc)
        latencies = [item.latency_ms for item in self.results]
        approval_cases = [
            item for item in self.results if item.case_id in {"observation-approval-execute", "versioned-reference"}
        ]
        unnecessary_checks = [
            check
            for item in self.results
            for check in item.checks
            if check.name.startswith("unnecessary_tool:")
        ]
        schema_chars = [
            value for item in self.results for value in item.tool_schema_chars_per_turn
        ]
        observation_chars = [
            value for item in self.results for value in item.observation_view_chars_per_turn
        ]
        context_chars = [
            value for item in self.results for value in item.conversation_context_chars_per_turn
        ]
        passed = sum(item.passed for item in self.results)
        pass_rate = passed / len(self.results) if self.results else 0.0
        p95 = _percentile(latencies, 0.95)
        approval_rate = (
            sum(
                any(check.name == "approval_integrity" and check.passed for check in item.checks)
                for item in approval_cases
            )
            / len(approval_cases)
            if approval_cases
            else 1.0
        )
        unnecessary_rate = (
            sum(not check.passed for check in unnecessary_checks) / len(unnecessary_checks)
            if unnecessary_checks
            else 0.0
        )
        slo_passed = (
            pass_rate >= 0.90
            and p95 <= self.p95_slo_ms
            and approval_rate == 1.0
            and idempotency_passed
            and concurrency_passed
        )
        return ProductionOnlineReport(
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            environment={
                "model": f"live:{settings.model_name}",
                "database": "live:postgresql",
                "checkpoint": "live:postgresql",
                "rag": "live:local-hybrid-plus-embedding",
                "http_boundary": "real:fastapi-asgi",
                "data": "synthetic-only",
                "auth": "excluded",
                "privacy_gateway": "excluded",
                "google_drive": "excluded",
                "weather": "excluded",
            },
            thresholds={
                "minimum_pass_rate": 0.90,
                "maximum_p95_latency_ms": self.p95_slo_ms,
                "minimum_approval_safety_rate": 1.0,
                "minimum_concurrency_success_rate": 1.0,
            },
            summary=ProductionOnlineSummary(
                total=len(self.results),
                passed=passed,
                pass_rate=pass_rate,
                p50_latency_ms=_percentile(latencies, 0.50),
                p95_latency_ms=p95,
                max_latency_ms=max(latencies, default=0.0),
                approval_safety_rate=approval_rate,
                idempotency_passed=idempotency_passed,
                concurrency_passed=concurrency_passed,
                unnecessary_tool_call_rate=unnecessary_rate,
                average_tool_schema_chars_per_turn=mean(schema_chars) if schema_chars else 0.0,
                average_observation_view_chars_per_turn=(
                    mean(observation_chars) if observation_chars else 0.0
                ),
                average_conversation_context_chars_per_turn=(
                    mean(context_chars) if context_chars else 0.0
                ),
                slo_passed=slo_passed,
            ),
            results=self.results,
        )

    async def _runtime_factory(self) -> ApiRuntime:
        """Real runtime with synthetic-only conversation context and no memory writes."""

        store = AsyncEasyTeachingStore(settings.database_url)
        await store.initialize()
        try:
            checkpointer = await build_postgres_checkpointer(
                settings.checkpoint_database_url
            )
            registry = build_default_tool_registry(store)
            context_reader = _SyntheticContextReader(store)
            graph = build_main_react_graph(
                checkpointer=checkpointer,
                registry=registry,
                long_memory_store=store,
                long_memory_extractor=_NoOpMemoryExtractor(),
                context_manager=ContextManager(
                    long_term_memory_reader=context_reader
                ),
            )
        except Exception:
            await store.close()
            raise
        return ApiRuntime(
            store=store,
            checkpointer=checkpointer,
            graph=graph,
            tool_registry=registry,
            privacy_gateway_mode="disabled",
        )

    def _new_session(self, client: TestClient) -> Dict[str, Any]:
        response = client.post(
            "/sessions",
            json={"teacher_id": "teacher-001", "class_id": "kangaroo-room"},
        )
        response.raise_for_status()
        return response.json()

    def _message(
        self,
        client: TestClient,
        *,
        session_id: str,
        message: str,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        request_id = request_id or str(uuid4())
        started = perf_counter()
        accepted = client.post(
            f"/sessions/{session_id}/messages",
            json={"message": message, "request_id": request_id},
        )
        latency_ms = (perf_counter() - started) * 1000
        draft = client.get(f"/sessions/{session_id}/drafts/{request_id}")
        events_response = client.get(
            f"/sessions/{session_id}/events",
            params={"request_id": request_id},
        )
        events = parse_sse_events(events_response.text) if events_response.status_code == 200 else []
        return {
            "request_id": request_id,
            "accepted": accepted,
            "draft_response": draft,
            "draft": draft.json() if draft.status_code == 200 else {},
            "events": events,
            "latency_ms": latency_ms,
            **_trace_metrics(events),
        }

    def _record(
        self,
        case_id: str,
        run: Dict[str, Any],
        checks: List[ProductionCheck],
    ) -> ProductionCaseResult:
        draft_payload = run.get("draft") or {}
        draft = draft_payload.get("draft") or {}
        result = ProductionCaseResult(
            case_id=case_id,
            passed=all(check.passed for check in checks),
            latency_ms=run.get("latency_ms", 0.0),
            status=str(draft_payload.get("status") or run.get("status") or "unknown"),
            tools=run.get("tools", []),
            citation_count=len(draft_payload.get("citations") or []),
            answer_preview=" ".join(str(draft.get("content") or "").split())[:300],
            checks=checks,
            tool_schema_chars_per_turn=run.get("tool_schema_chars_per_turn", []),
            observation_view_chars_per_turn=run.get("observation_view_chars_per_turn", []),
            conversation_context_chars_per_turn=run.get(
                "conversation_context_chars_per_turn", []
            ),
        )
        self.results.append(result)
        return result

    def _health_and_validation(self, client: TestClient) -> None:
        started = perf_counter()
        health = client.get("/health")
        session = self._new_session(client)
        invalid = client.post(
            f"/sessions/{session['session_id']}/messages",
            json={"message": "Synthetic", "unexpected": True},
        )
        latency = (perf_counter() - started) * 1000
        self._record(
            "http-health-validation",
            {"latency_ms": latency, "status": "completed"},
            [
                _check("health", 200, health.status_code),
                _check("unknown_field_rejected", 422, invalid.status_code),
            ],
        )

    def _activity(self, client: TestClient) -> None:
        session = self._new_session(client)
        run = self._message(
            client,
            session_id=session["session_id"],
            message=(
                "Create a short sensory activity for the current Kangaroo Room. "
                "Use class context and perform the activity safety check. Do not use "
                "EYLF, NQS, prior records, Drive or save anything."
            ),
        )
        tools = run["tools"]
        self._record(
            "activity-class-safety",
            run,
            [
                _check("status", "completed", run["draft"].get("status")),
                _check("get_class_context", True, "get_class_context" in tools),
                _check("check_activity_safety", True, "check_activity_safety" in tools),
                _check("unnecessary_tool:retrieve_knowledge", False, "retrieve_knowledge" in tools),
                _check("no_approval", "not_required", (run["draft"].get("approval") or {}).get("status")),
            ],
        )

    def _policy(self, client: TestClient) -> None:
        session = self._new_session(client)
        run = self._message(
            client,
            session_id=session["session_id"],
            message=(
                "Using only EYLF, explain how play-based learning supports agency. "
                "Cite the retrieved EYLF evidence and do not save anything."
            ),
        )
        answer = str((run["draft"].get("draft") or {}).get("content") or "")
        self._record(
            "policy-eylf-rag",
            run,
            [
                _check("status", "completed", run["draft"].get("status")),
                _check("retrieve_knowledge", True, "retrieve_knowledge" in run["tools"]),
                _check("citations", ">=1", len(run["draft"].get("citations") or []), len(run["draft"].get("citations") or []) >= 1),
                _check("answer_mentions_eylf", True, "eylf" in answer.casefold()),
                _check("unnecessary_tool:check_activity_safety", False, "check_activity_safety" in run["tools"]),
            ],
        )

    def _observation_write_and_query(self, client: TestClient) -> None:
        session = self._new_session(client)
        run = self._message(
            client,
            session_id=session["session_id"],
            message=(
                f"Save this complete synthetic observation as a draft: on 25 August "
                f"2026 at 10:15 am during indoor play, a child arranged four blocks "
                f"beside the marker {self.marker} and counted them aloud. No child link "
                "is required."
            ),
        )
        approval = run["draft"].get("approval") or {}
        preview = approval.get("preview") or {}
        approve = client.post(
            f"/sessions/{session['session_id']}/approvals",
            json={"request_id": run["request_id"], "decision": "approve"},
        )
        completed = client.get(
            f"/sessions/{session['session_id']}/drafts/{run['request_id']}"
        )
        self._record(
            "observation-approval-execute",
            run,
            [
                _check("waiting", "waiting_for_approval", run["draft"].get("status")),
                _check("save_tool", "save_observation", approval.get("tool_name")),
                _check("preview_marker", True, self.marker.casefold() in str(preview).casefold()),
                _check("approval_integrity", True, bool(approval.get("action_id") and preview)),
                _check("approval_http", 200, approve.status_code),
                _check("executed", "completed", completed.json().get("status") if completed.status_code == 200 else None),
            ],
        )

        query_session = self._new_session(client)
        query = self._message(
            client,
            session_id=query_session["session_id"],
            message=(
                f"Find the saved observation containing {self.marker}. Report only "
                "documented facts and do not save or export anything."
            ),
        )
        answer = str((query["draft"].get("draft") or {}).get("content") or "")
        self._record(
            "observation-query-after-write",
            query,
            [
                _check("status", "completed", query["draft"].get("status")),
                _check("query_records", True, "query_records" in query["tools"]),
                _check("saved_marker_returned", True, self.marker.casefold() in answer.casefold()),
            ],
        )

    def _versioned_reference(self, client: TestClient) -> None:
        session = self._new_session(client)
        first = f"{self.marker}-LEAF-V1"
        second = f"{self.marker}-WATER-V2"
        self._message(
            client,
            session_id=session["session_id"],
            message=(
                f"Create an activity draft titled {first} using leaves and bark. "
                "Do not search policy or records and do not save it."
            ),
        )
        self._message(
            client,
            session_id=session["session_id"],
            message=(
                f"Create a different second activity titled {second} using pouring "
                "containers. Keep the first version and do not save either."
            ),
        )
        run = self._message(
            client,
            session_id=session["session_id"],
            message=f"Save the first version, {first}, as a draft educational record.",
        )
        approval = run["draft"].get("approval") or {}
        preview_text = str(approval.get("preview") or {}).casefold()
        reject = client.post(
            f"/sessions/{session['session_id']}/approvals",
            json={"request_id": run["request_id"], "decision": "reject"},
        )
        self._record(
            "versioned-reference",
            run,
            [
                _check("waiting", "waiting_for_approval", run["draft"].get("status")),
                _check("save_tool", "save_educational_record", approval.get("tool_name")),
                _check("selected_first", True, first.casefold() in preview_text),
                _check("did_not_select_second", False, second.casefold() in preview_text),
                _check("approval_integrity", True, bool(approval.get("action_id") and approval.get("preview"))),
                _check("reject_http", 200, reject.status_code),
            ],
        )

    def _cross_framework(self, client: TestClient) -> None:
        session = self._new_session(client)
        run = self._message(
            client,
            session_id=session["session_id"],
            message=(
                "Compare EYLF and NQS evidence for intentional teaching and child "
                "agency. Retrieve each source boundary and cite both."
            ),
        )
        answer = str((run["draft"].get("draft") or {}).get("content") or "").casefold()
        sources = {
            str(item.get("source") or "")
            for item in run["draft"].get("citations") or []
        }
        direct_retrievals = run["tools"].count("retrieve_knowledge")
        worker_research_paths = sum(
            name.endswith("_worker") for name in run["tools"]
        )
        research_paths = max(direct_retrievals, worker_research_paths)
        self._record(
            "cross-framework-rag",
            run,
            [
                _check("status", "completed", run["draft"].get("status")),
                _check(
                    "two_research_paths",
                    ">=2",
                    research_paths,
                    research_paths >= 2,
                ),
                _check("two_sources", ">=2", len(sources), len(sources) >= 2),
                _check("answer_eylf", True, "eylf" in answer),
                _check(
                    "answer_nqs_or_nqf",
                    True,
                    any(
                        marker in answer
                        for marker in (
                            "nqs",
                            "national quality standard",
                            "nqf",
                            "national quality framework",
                        )
                    ),
                ),
            ],
        )


    def _idempotency(self, client: TestClient) -> bool:
        session = self._new_session(client)
        request_id = str(uuid4())
        first = self._message(
            client,
            session_id=session["session_id"],
            request_id=request_id,
            message=f"Organise this synthetic note without saving: {self.marker}-IDEMPOTENT.",
        )
        second_started = perf_counter()
        second = client.post(
            f"/sessions/{session['session_id']}/messages",
            json={"message": "This changed text must not rerun.", "request_id": request_id},
        )
        second_latency = (perf_counter() - second_started) * 1000
        run_started_count = sum(event.get("event") == "run_started" for event in first["events"])
        checks = [
            _check("first_completed", "completed", first["draft"].get("status")),
            _check("retry_http", 202, second.status_code),
            _check("same_request", request_id, second.json().get("request_id")),
            _check("single_run_started", 1, run_started_count),
            _check("fast_retry", "<1000ms", round(second_latency), second_latency < 1000),
        ]
        self._record("request-idempotency", first, checks)
        return all(check.passed for check in checks)

    def _concurrency(self, client: TestClient) -> bool:
        sessions = [self._new_session(client) for _ in range(self.concurrency)]

        def submit(index: int) -> Dict[str, Any]:
            marker = f"{self.marker}-CONCURRENT-{index}"
            run = self._message(
                client,
                session_id=sessions[index]["session_id"],
                message=(
                    f"Organise this as a concise objective note, do not use tools and "
                    f"do not save it: a child placed three rings in a row beside {marker}."
                ),
            )
            run["expected_marker"] = marker
            return run

        started = perf_counter()
        errors: List[str] = []
        runs: List[Dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = [pool.submit(submit, index) for index in range(self.concurrency)]
                for future in futures:
                    try:
                        runs.append(future.result())
                    except Exception as error:  # pragma: no cover - live environment
                        errors.append(f"{type(error).__name__}: {error}")
        except Exception as error:  # pragma: no cover - live environment
            errors.append(f"{type(error).__name__}: {error}")
        wall_ms = (perf_counter() - started) * 1000
        successful = 0
        for run in runs:
            answer = str((run["draft"].get("draft") or {}).get("content") or "")
            if (
                run["draft"].get("status") == "completed"
                and run["expected_marker"].casefold() in answer.casefold()
            ):
                successful += 1
        checks = [
            _check("concurrent_successes", self.concurrency, successful),
            _check("concurrent_errors", [], errors),
        ]
        self._record(
            "concurrent-independent-sessions",
            {"latency_ms": wall_ms, "status": "completed" if not errors else "failed"},
            checks,
        )
        return all(check.passed for check in checks)


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


class _SyntheticContextReader:
    """Expose only this suite's conversation artifacts, never stored profile memory."""

    def __init__(self, store: AsyncEasyTeachingStore) -> None:
        self.store = store

    def list_profile_memories(self, **_: Any) -> List[Any]:
        return []

    async def get_conversation_workspace(self, **kwargs: Any) -> Dict[str, Any]:
        return await self.store.get_conversation_workspace(**kwargs)


class _NoOpMemoryExtractor:
    async def decide_async(self, **_: Any) -> List[Any]:
        return []


def format_production_report(report: ProductionOnlineReport) -> str:
    summary = report.summary
    lines = [
        "=== EasyTeaching Production Online Evaluation ===",
        f"cases={summary.total} passed={summary.passed} pass_rate={summary.pass_rate:.1%} slo_passed={summary.slo_passed}",
        f"latency_p50={summary.p50_latency_ms:.0f}ms latency_p95={summary.p95_latency_ms:.0f}ms max={summary.max_latency_ms:.0f}ms",
        f"approval_safety={summary.approval_safety_rate:.1%} idempotency={summary.idempotency_passed} concurrency={summary.concurrency_passed} unnecessary_tool_calls={summary.unnecessary_tool_call_rate:.1%}",
        f"avg_tool_schema_chars={summary.average_tool_schema_chars_per_turn:.0f} avg_observation_chars={summary.average_observation_view_chars_per_turn:.0f} avg_context_chars={summary.average_conversation_context_chars_per_turn:.0f}",
    ]
    failed = [item for item in report.results if not item.passed]
    if failed:
        lines.append("Failed cases:")
        for item in failed:
            failed_checks = ", ".join(check.name for check in item.checks if not check.passed)
            lines.append(f"  {item.case_id}: {failed_checks}")
    return "\n".join(lines)


def run_production_online_eval(
    *,
    p95_slo_ms: float = 30_000,
    concurrency: int = 3,
    case_ids: Optional[List[str]] = None,
) -> ProductionOnlineReport:
    return ProductionOnlineRunner(
        p95_slo_ms=p95_slo_ms,
        concurrency=concurrency,
        case_ids=case_ids,
    ).run()
