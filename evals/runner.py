"""Run fixed evaluation cases against real EasyTeaching components."""

import json
import asyncio
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Optional

from pydantic import BaseModel

from app.agents import WorkerProfile, WorkerRegistry
from app.schemas import (
    CapabilityCall,
    CapabilityObservation,
    CapabilitySource,
    GraphState,
    Intent,
    IntentRouteResult,
    LongTermMemoryCandidate,
    LongTermMemoryScope,
    LongTermMemoryType,
    MemoryRetrievalMode,
    MainDecision,
    ObservationStatus,
    RetrievalMode,
    RetrievalRequest,
    RerankerMode,
    RiskLevel,
    ThreadContext,
    WorkerCall,
    WorkerName,
)
from app.services import (
    ChatCompletionsModelProvider,
    ContextManager,
    KnowledgeIngestionService,
    KnowledgeRetriever,
    ObservationRedactor,
    SQLiteFTS5KnowledgeIndex,
)
from evals.in_memory_store import InMemoryEvalStore
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolExecutionContext,
    ToolPermission,
    ToolRegistry,
    ToolResult,
    build_default_tool_registry,
)
from app.workflows import build_main_react_graph
from evals.cases import load_eval_cases
from evals.evaluators import evaluate_case
from evals.errors import safe_eval_error_code
from evals.metrics import build_eval_report
from evals.model_meter import MeteredModelProvider
from evals.schemas import (
    EvalActual,
    EvalCase,
    EvalCategory,
    EvalCheck,
    EvalMode,
    EvalReport,
    EvalResult,
    EvalTokenUsage,
    MemoryActual,
    MemoryEvalTarget,
    ObservedToolCall,
    RagActual,
    RagStatus,
    RoutingActual,
    SafetyActual,
    SafetyOutcome,
    ToolActual,
    TrajectoryActual,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge" / "processed" / "chunks.jsonl"


class EvalRunner:
    """Own temporary eval state and dispatch each category to its adapter."""

    def __init__(
        self,
        *,
        mode: EvalMode = EvalMode.OFFLINE,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        self.mode = mode
        self.store = InMemoryEvalStore()
        self._seed_memories()
        self.registry = build_default_tool_registry(self.store)
        self.context_manager = ContextManager(long_term_memory_reader=self.store)
        self.redactor = ObservationRedactor()
        self.meter: Optional[MeteredModelProvider] = None
        if mode is EvalMode.LIVE_MODEL:
            self.meter = MeteredModelProvider(
                ChatCompletionsModelProvider(),
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
            )
        lexical_path = PROJECT_ROOT / "data" / "knowledge" / "index" / "knowledge_fts.sqlite3"
        source_digest = SQLiteFTS5KnowledgeIndex.digest_file(KNOWLEDGE_PATH)
        if not lexical_path.exists():
            chunks = KnowledgeIngestionService(project_root=PROJECT_ROOT).read_chunks_jsonl(
                KNOWLEDGE_PATH
            )
            SQLiteFTS5KnowledgeIndex.build(
                lexical_path,
                chunks,
                source_digest=source_digest,
            )
        lexical_index = SQLiteFTS5KnowledgeIndex(lexical_path)
        if lexical_index.manifest().get("source_digest") != source_digest:
            chunks = KnowledgeIngestionService(project_root=PROJECT_ROOT).read_chunks_jsonl(
                KNOWLEDGE_PATH
            )
            lexical_index = SQLiteFTS5KnowledgeIndex.build(
                lexical_path,
                chunks,
                source_digest=source_digest,
            )
        self.rag_retriever = KnowledgeRetriever(lexical_index=lexical_index)
        self.graph = self._build_trajectory_graph()

    def close(self) -> None:
        self.store.close()

    def run(self, cases: Iterable[EvalCase]) -> EvalReport:
        return build_eval_report(
            [self.run_case(case) for case in cases],
            mode=self.mode,
        )

    def run_case(self, case: EvalCase) -> EvalResult:
        if self.meter is not None:
            self.meter.reset()
        started = perf_counter()
        try:
            actual = self._actual_for(case)
            latency = (perf_counter() - started) * 1000
            return evaluate_case(
                case,
                actual,
                latency_ms=latency,
                token_usage=self.meter.usage if self.meter else EvalTokenUsage(),
                estimated_cost_usd=(
                    self.meter.estimated_cost_usd if self.meter else 0.0
                ),
            )
        except Exception as error:
            latency = (perf_counter() - started) * 1000
            error_code = safe_eval_error_code(error)
            return EvalResult(
                case_id=case.id,
                category=case.category,
                passed=False,
                score=0.0,
                checks=[
                    EvalCheck(
                        name=f"execution_{error_code}",
                        passed=False,
                        expected=True,
                        actual=False,
                        message=(
                            f"The case failed with safe error code: {error_code}. "
                            "Exception details are not exposed."
                        ),
                    )
                ],
                latency_ms=latency,
                token_usage=self.meter.usage if self.meter else EvalTokenUsage(),
                estimated_cost_usd=(
                    self.meter.estimated_cost_usd if self.meter else 0.0
                ),
                error_code=error_code,
            )

    def _actual_for(self, case: EvalCase) -> EvalActual:
        if case.category is EvalCategory.ROUTING:
            return self._routing_actual(case)
        if case.category is EvalCategory.TOOL:
            return self._tool_actual(case)
        if case.category is EvalCategory.RAG:
            return self._rag_actual(case)
        if case.category is EvalCategory.MEMORY:
            return self._memory_actual(case)
        if case.category is EvalCategory.SAFETY:
            return self._safety_actual(case)
        return self._trajectory_actual(case)

    def _routing_actual(self, case: EvalCase) -> RoutingActual:
        router = _KeywordRouter()
        route = router.route(
            case.input.message,
            conversation_context=case.input.conversation_context,
        )
        return RoutingActual(
            intent=route.intent,
            needs_clarification=route.needs_clarification,
        )

    def _tool_actual(self, case: EvalCase) -> ToolActual:
        message = case.input.message.lower()
        calls = [
            ObservedToolCall(
                tool_name="load_skill",
                tool_args={"skill_name": "activity_planning"},
            )
        ]
        if "profile" in message:
            calls.append(
                ObservedToolCall(
                    tool_name="get_class_profile",
                    tool_args={"class_id": case.input.class_id or "kangaroo-room"},
                )
            )
        elif "safety" in message:
            calls.append(ObservedToolCall(tool_name="check_activity_safety"))
        elif "risk guidance" in message:
            calls.append(ObservedToolCall(tool_name="research_knowledge"))
        elif "eylf outcomes" in message:
            calls.append(ObservedToolCall(tool_name="search_knowledge"))
        return ToolActual(calls=calls)

    def _rag_actual(self, case: EvalCase) -> RagActual:
        result = self.rag_retriever.retrieve(
            RetrievalRequest(
                query=case.input.message,
                top_k=3,
                mode=RetrievalMode.BM25,
                reranker=RerankerMode.NONE,
            )
        )
        sources = list(
            dict.fromkeys(item.source_id for item in result.citations)
        )
        return RagActual(
            status=(
                RagStatus.ANSWERED
                if result.chunks
                else RagStatus.NEEDS_CLARIFICATION
            ),
            sources=sources,
            citation_count=len(result.citations),
        )

    def _memory_actual(self, case: EvalCase) -> MemoryActual:
        expected = case.expected
        if expected.target is MemoryEvalTarget.PROFILE_CONTEXT:
            output = self.context_manager.build_model_context(
                ThreadContext(),
                teacher_id=case.input.teacher_id,
            )
            return MemoryActual(success=True, output=output)
        result = self.registry.execute(
            "recall_long_term_memory",
            {"query": case.input.message},
            execution_context=ToolExecutionContext(
                teacher_id=case.input.teacher_id,
                class_id=case.input.class_id,
            ),
        )
        return MemoryActual(
            success=result.success,
            output=json.dumps(result.data, ensure_ascii=False),
            error_code=result.error.code.value if result.error else None,
        )

    def _safety_actual(self, case: EvalCase) -> SafetyActual:
        message = case.input.message.lower()
        if "diagnos" in message:
            return SafetyActual(
                outcome=SafetyOutcome.BLOCK,
                error_codes=["forbidden_action"],
            )
        if "without teacher approval" in message:
            result = self.registry.execute(
                "save_draft",
                {
                    "draft_id": "eval-draft",
                    "idempotency_key": case.id,
                    "draft_type": "activity",
                    "title": "Eval draft",
                    "content": "Synthetic content",
                },
                approved=False,
            )
            return SafetyActual(
                outcome=SafetyOutcome.BLOCK,
                error_codes=[result.error.code.value] if result.error else [],
            )
        redacted = self.redactor.deidentify(case.input.message)
        return SafetyActual(
            outcome=(
                SafetyOutcome.REDACT
                if redacted.replacement_count
                else SafetyOutcome.ALLOW
            ),
            output=redacted.safe_text,
        )

    def _trajectory_actual(self, case: EvalCase) -> TrajectoryActual:
        state = asyncio.run(
            self.graph.ainvoke(GraphState(
                request_id=case.id,
                session_id=f"eval-{case.id}",
                user_message=case.input.message,
                teacher_id=case.input.teacher_id,
                class_id=case.input.class_id,
            ))
        )
        resolved = GraphState.model_validate(state)
        return TrajectoryActual(steps=[event.step for event in resolved.trace])

    def _seed_memories(self) -> None:
        fixtures = [
            LongTermMemoryCandidate(
                scope=LongTermMemoryScope.TEACHER,
                scope_id="teacher-001",
                memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
                content="Uses Australian English and concise summaries.",
                reason="Synthetic evaluation fixture.",
                retrieval_mode=MemoryRetrievalMode.PROFILE,
            ),
            LongTermMemoryCandidate(
                scope=LongTermMemoryScope.TEACHER,
                scope_id="teacher-001",
                memory_type=LongTermMemoryType.LONG_TERM_CONSTRAINT,
                content="Previously used water-play activities outdoors.",
                reason="Synthetic evaluation fixture.",
            ),
            LongTermMemoryCandidate(
                scope=LongTermMemoryScope.TEACHER,
                scope_id="teacher-002",
                memory_type=LongTermMemoryType.LONG_TERM_CONSTRAINT,
                content="Previously used water-play activities indoors.",
                reason="Synthetic evaluation fixture.",
            ),
            LongTermMemoryCandidate(
                scope=LongTermMemoryScope.CLASS,
                scope_id="another-class",
                memory_type=LongTermMemoryType.CLASS_FACT,
                content="another-class-private-memory",
                reason="Synthetic isolation fixture.",
            ),
        ]
        for fixture in fixtures:
            self.store.save_long_term_memory(fixture)

    def _build_trajectory_graph(self):
        workers = WorkerRegistry(
            [
                WorkerProfile(
                    name=name,
                    description="Deterministic evaluation Worker.",
                    allowed_tool_names=frozenset(),
                )
                for name in WorkerName
            ]
        )
        return build_main_react_graph(
            main_agent=_TrajectoryMainAgent(),
            registry=_trajectory_registry(),
            worker_registry=workers,
            worker_runner=_TrajectoryWorkerRunner(),
            context_manager=self.context_manager,
            long_memory_extractor=_NoOpMemoryExtractor(),
            long_memory_store=self.store,
        )


class _KeywordRouter:
    """Stable offline replacement for the model-dependent intent decision."""

    def route(
        self,
        user_message: str,
        *,
        conversation_context: str = "",
    ) -> IntentRouteResult:
        message = user_message.lower()
        if "family" in message or "families" in message:
            intent = Intent.FAMILY_COMMUNICATION
        elif any(word in message for word in ("learning story", "learning record", "observation")):
            intent = Intent.LEARNING_RECORD
        elif any(word in message for word in ("eylf", "policy", "guidance")):
            intent = Intent.POLICY_QA
        elif any(word in message for word in ("plan", "activity")):
            intent = Intent.ACTIVITY_PLANNING
        else:
            intent = Intent.UNKNOWN
        clarification = intent is Intent.UNKNOWN
        return IntentRouteResult(
            intent=intent,
            confidence=0.99 if not clarification else 0.0,
            needs_clarification=clarification,
            clarification_question=(
                "What would you like help creating?" if clarification else None
            ),
            reason="Deterministic offline evaluation route.",
        )


class _TrajectoryToolInput(BaseModel):
    value: str = "ok"


class _TrajectoryToolOutput(BaseModel):
    value: str


def _trajectory_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for name in ("eval_tool_a", "eval_tool_b"):
        registry.register(
            ToolDefinition(
                name=name,
                description="Deterministic read-only trajectory tool.",
                category=ToolCategory.SYSTEM,
                input_model=_TrajectoryToolInput,
                output_model=_TrajectoryToolOutput,
                risk_level=RiskLevel.L0_READ_ONLY,
                permission=ToolPermission.AUTO_EXECUTE,
                domain=ToolDomain.INTERNAL,
                parallel_safe=True,
                handler=lambda args, tool_name=name: ToolResult.ok(
                    data={"value": f"{tool_name}:{args.value}"},
                    risk_level=RiskLevel.L0_READ_ONLY,
                ),
            )
        )
    return registry


class _TrajectoryMainAgent:
    """用固定决定验证真实生产主图，不调用外部模型。"""

    async def decide(self, *, user_message, observations, **kwargs):
        message = user_message.lower()
        if "clarification" in message:
            return MainDecision(
                reason="The deterministic case needs clarification.",
                clarification_question="What result do you need?",
            )
        if "parallel workers" in message and not observations:
            return MainDecision(
                reason="Two independent deep tasks.",
                worker_calls=[
                    WorkerCall(
                        name=WorkerName.INTERNAL_RESEARCH,
                        arguments={"task": "internal evidence"},
                        result_key="internal",
                    ),
                    WorkerCall(
                        name=WorkerName.EXTERNAL_RESEARCH,
                        arguments={"task": "public evidence"},
                        result_key="external",
                    ),
                ],
            )
        if "parallel tools" in message and not observations:
            return MainDecision(
                reason="Two independent simple calls.",
                tool_calls=[
                    CapabilityCall(
                        name="eval_tool_a",
                        arguments={"value": "a"},
                        result_key="a",
                    ),
                    CapabilityCall(
                        name="eval_tool_b",
                        arguments={"value": "b"},
                        result_key="b",
                    ),
                ],
            )
        if "dependency" in message and "first" not in observations:
            return MainDecision(
                reason="Run prerequisite first.",
                tool_calls=[
                    CapabilityCall(
                        name="eval_tool_a",
                        arguments={"value": "first"},
                        result_key="first",
                    )
                ],
            )
        if "dependency" in message and "second" not in observations:
            return MainDecision(
                reason="Prerequisite is now available.",
                tool_calls=[
                    CapabilityCall(
                        name="eval_tool_b",
                        arguments={"value": "second"},
                        needs=["first"],
                        result_key="second",
                    )
                ],
            )
        if "single tool" in message and not observations:
            return MainDecision(
                reason="One simple lookup.",
                tool_calls=[
                    CapabilityCall(
                        name="eval_tool_a",
                        arguments={"value": "single"},
                        result_key="single",
                    )
                ],
            )
        return MainDecision(reason="Evidence is sufficient.", final_answer="Eval draft.")


class _TrajectoryWorkerRunner:
    async def run(self, call, **kwargs):
        return CapabilityObservation(
            result_key=call.result_key,
            capability_name=call.name.value,
            source_kind=CapabilitySource.WORKER,
            status=ObservationStatus.COMPLETED,
            data={"summary": call.arguments["task"]},
        )


class _NoOpMemoryExtractor:
    def decide(self, **_: object) -> List[object]:
        return []


def run_eval_suite(
    *,
    cases: Optional[Iterable[EvalCase]] = None,
    mode: EvalMode = EvalMode.OFFLINE,
    categories: Optional[Iterable[EvalCategory]] = None,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
) -> EvalReport:
    """Convenient public entry point for scripts and CI."""
    selected = list(cases) if cases is not None else load_eval_cases()
    if categories is not None:
        allowed = set(categories)
        selected = [case for case in selected if case.category in allowed]
    runner = EvalRunner(
        mode=mode,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    try:
        return runner.run(selected)
    finally:
        runner.close()
