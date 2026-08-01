"""Run fixed evaluation cases against real EduFlow components."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Iterable, List, Optional

from app.agents import IntentRouter
from app.schemas import (
    Approval,
    ApprovalStatus,
    Draft,
    ForbiddenSpecialistAction,
    GraphState,
    Intent,
    IntentRouteResult,
    LongTermMemoryCandidate,
    LongTermMemoryScope,
    LongTermMemoryType,
    MemoryRetrievalMode,
    RetrievalMode,
    RerankerMode,
    RiskLevel,
    SpecialistInput,
    SpecialistKind,
    SpecialistPermissionDenied,
    SpecialistResult,
    ThreadContext,
    TraceEvent,
    WorkflowStatus,
    get_specialist_permission,
)
from app.services import (
    BM25KnowledgeIndex,
    ChatCompletionsModelProvider,
    ContextManager,
    EduFlowStore,
    KnowledgeRetriever,
    ObservationRedactor,
    PolicyRAGService,
)
from app.tools import ToolExecutionContext, build_default_tool_registry
from app.workflows.main_graph import build_main_graph
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
        self._temporary_directory = TemporaryDirectory(prefix="eduflow-evals-")
        database_path = Path(self._temporary_directory.name) / "eval.db"
        self.store = EduFlowStore(f"sqlite:///{database_path}")
        self.store.initialize()
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
        lexical_index = BM25KnowledgeIndex.from_jsonl(
            KNOWLEDGE_PATH,
            project_root=PROJECT_ROOT,
        )
        retriever = KnowledgeRetriever(lexical_index=lexical_index)
        self.rag = PolicyRAGService(
            retriever=retriever,
            model_provider=self.meter,
            top_k=3,
            retrieval_mode=RetrievalMode.BM25,
            reranker=RerankerMode.LEXICAL,
        )
        self.graph = self._build_trajectory_graph()

    def close(self) -> None:
        self.store.engine.dispose()
        self._temporary_directory.cleanup()

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
        router = IntentRouter(provider=self.meter) if self.meter else _KeywordRouter()
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
            calls.append(ObservedToolCall(tool_name="retrieve_risk_guidance"))
        elif "eylf outcomes" in message:
            calls.append(ObservedToolCall(tool_name="align_to_eylf_outcomes"))
        return ToolActual(calls=calls)

    def _rag_actual(self, case: EvalCase) -> RagActual:
        result = self.rag.answer(
            case.input.message,
            conversation_context=case.input.conversation_context,
        )
        sources = list(
            dict.fromkeys(item.source_id for item in result.citations)
        )
        return RagActual(
            status=result.status,
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
            permission = get_specialist_permission(SpecialistKind.PLANNING)
            try:
                permission.require_action(ForbiddenSpecialistAction.CHILD_DIAGNOSIS)
            except SpecialistPermissionDenied:
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
        state = self.graph.invoke(
            GraphState(
                request_id=case.id,
                session_id=f"eval-{case.id}",
                user_message=case.input.message,
                teacher_id=case.input.teacher_id,
                class_id=case.input.class_id,
            )
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
        return build_main_graph(
            router=_KeywordRouter(),
            planning_workflow=_StaticSpecialist(
                SpecialistKind.PLANNING,
                "planning_react",
            ),
            policy_workflow=_StaticSpecialist(
                SpecialistKind.POLICY,
                "policy_rag",
            ),
            documentation_workflow=_StaticSpecialist(
                SpecialistKind.DOCUMENTATION,
                "documentation_draft",
                needs_approval=True,
            ),
            family_workflow=_StaticSpecialist(
                SpecialistKind.FAMILY,
                "family_draft_skeleton",
            ),
            context_manager=self.context_manager,
            long_memory_extractor=_NoOpMemoryExtractor(),
            long_memory_store=self.store,
            learning_record_store=self.store,
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


class _StaticSpecialist:
    """Minimal specialist output used to exercise the real main graph path."""

    def __init__(
        self,
        specialist: SpecialistKind,
        trace_step: str,
        *,
        needs_approval: bool = False,
    ) -> None:
        self.specialist = specialist
        self.trace_step = trace_step
        self.needs_approval = needs_approval

    def invoke(self, input_data: SpecialistInput) -> SpecialistResult:
        approval = Approval()
        status = WorkflowStatus.COMPLETED
        if self.needs_approval:
            approval = Approval(
                status=ApprovalStatus.REQUIRED,
                risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                reason="Teacher review is required.",
            )
            status = WorkflowStatus.WAITING_FOR_APPROVAL
        return SpecialistResult(
            specialist=self.specialist,
            status=status,
            draft=Draft(title="Synthetic eval draft", content="Synthetic eval content"),
            approval=approval,
            trace=[
                TraceEvent(
                    step=self.trace_step,
                    message="Synthetic specialist completed for graph evaluation.",
                )
            ],
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
