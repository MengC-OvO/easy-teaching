"""EduFlow 生产用统一 Main ReAct LangGraph。"""

from typing import Any, Dict, List, Mapping, Optional, Protocol, Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from app.agents import (
    BoundedWorkerRunner,
    DEFAULT_WORKER_PROFILES,
    ExecutionRoute,
    MainDecisionValidator,
    MainReActAgent,
    MainToolExecutor,
    WorkerRegistry,
)
from app.schemas import (
    Approval,
    CapabilityObservation,
    Draft,
    Citation,
    GraphError,
    GraphState,
    MainDecision,
    ObservationStatus,
    TraceEvent,
    WorkerCall,
    WorkflowStatus,
)
from app.services import (
    ChatCompletionsModelProvider,
    ContextManager,
    AsyncEduFlowStore,
    LLMLongTermMemoryExtractor,
    ModelProviderError,
)
from app.tools import ToolPermission, ToolRegistry, build_default_tool_registry
from app.workflows.main_react_support import (
    ContextManagerProtocol,
    LongTermMemoryExtractorProtocol,
    LongTermMemoryStoreProtocol,
    build_context_update_node,
    build_long_memory_update_node,
    initialize,
)


GraphStateInput = Union[GraphState, Mapping[str, Any]]


class MainAgentProtocol(Protocol):
    async def decide(self, **kwargs) -> MainDecision:
        ...


class WorkerRunnerProtocol(Protocol):
    async def run(self, call: WorkerCall, **kwargs) -> CapabilityObservation:
        ...


def _state(state: GraphStateInput) -> GraphState:
    if isinstance(state, GraphState):
        return state
    return GraphState.model_validate(state)


def build_main_react_node(
    agent: MainAgentProtocol,
    registry: ToolRegistry,
    worker_registry: WorkerRegistry,
    context_manager: ContextManagerProtocol,
    *,
    max_steps: int,
):
    async def main_react(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        if current.react_step >= max_steps:
            return {
                "decision": MainDecision(
                    reason="Main ReAct reached its bounded step limit.",
                    final_answer=_bounded_fallback(current),
                ),
                "trace": [
                    TraceEvent(
                        step="main_react",
                        message="Main ReAct reached the maximum step budget.",
                        metadata={"current_step": current.react_step},
                    )
                ],
            }

        tools = [
            tool
            for tool in registry.list_tools()
            if tool.permission is ToolPermission.AUTO_EXECUTE
        ]
        try:
            context_builder = getattr(context_manager, "build_model_context_async", None)
            conversation_context = (
                await context_builder(
                    current.context,
                    teacher_id=current.teacher_id,
                )
                if context_builder is not None
                else context_manager.build_model_context(
                    current.context,
                    teacher_id=current.teacher_id,
                )
            )
            decision = await agent.decide(
                    user_message=current.user_message,
                    conversation_context=conversation_context,
                    observations=current.observations,
                    available_tools=tools,
                    available_workers=worker_registry.public_descriptions(),
                    current_step=current.react_step,
                    max_steps=max_steps,
            )
            return {
                "decision": decision,
                "trace": [
                    TraceEvent(
                        step="main_react",
                        message="Main selected the current executable action.",
                        metadata={"current_step": current.react_step},
                    )
                ],
            }
        except (ModelProviderError, TypeError, ValueError) as error:
            return {
                "decision": MainDecision(
                    reason="The model decision was unavailable.",
                    clarification_question=(
                        "I could not safely decide the next step. Could you briefly "
                        "restate what draft or information you need?"
                    ),
                ),
                "errors": [
                    GraphError(
                        code="main_react_model_error",
                        message="Main ReAct used a safe clarification fallback.",
                        recoverable=True,
                    )
                ],
                "trace": [
                    TraceEvent(
                        step="main_react",
                        message="Main ReAct used the clarification fallback.",
                    )
                ],
            }

    return main_react


def build_validate_decision_node(
    validator: MainDecisionValidator,
    *,
    max_tool_calls: int,
    max_worker_batches: int,
    max_workers_per_batch: int,
):
    def validate_decision(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        assert current.decision is not None
        decision = current.decision

        feedback = None
        if current.tool_call_count + len(decision.tool_calls) > max_tool_calls:
            feedback = validator.feedback("工具调用已达到本次请求预算。")
        elif decision.worker_calls and current.worker_batch_count >= max_worker_batches:
            feedback = validator.feedback("Worker 批次已达到本次请求预算。")
        elif len(decision.worker_calls) > max_workers_per_batch:
            feedback = validator.feedback("一个批次包含过多 Worker。")

        validation = feedback or validator.validate(
            decision,
            observations=current.observations,
            repeated_call_counts=current.repeated_call_counts,
        )
        return {
            "execution_route": validation.route.value,
            "validation_feedback": validation.feedback,
            "trace": [
                TraceEvent(
                    step="validate_decision",
                    message="Validated the current Main decision.",
                    metadata={"status": validation.route.value},
                )
            ],
        }

    return validate_decision


def route_validated_decision(state: GraphStateInput):
    current = _state(state)
    route = ExecutionRoute(current.execution_route)
    if route is ExecutionRoute.PARALLEL_WORKERS:
        assert current.decision is not None
        return [
            Send(
                "run_worker",
                {
                    "call": call.model_dump(mode="json"),
                    "teacher_id": current.teacher_id,
                    "class_id": current.class_id,
                    "dependency_observations": {
                        key: current.observations[key].model_dump(mode="json")
                        for key in call.needs
                    },
                },
            )
            for call in current.decision.worker_calls
        ]
    return route.value


def build_single_tool_node(executor: MainToolExecutor):
    async def single_tool(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        assert current.decision is not None
        observation = await executor.execute_one(
                current.decision.tool_calls[0],
                teacher_id=current.teacher_id,
                class_id=current.class_id,
        )
        return {"pending_observations": [observation]}

    return single_tool


def build_parallel_tools_node(executor: MainToolExecutor):
    async def parallel_tools(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        assert current.decision is not None
        observations = await executor.execute_many(
                current.decision.tool_calls,
                teacher_id=current.teacher_id,
                class_id=current.class_id,
        )
        return {"pending_observations": observations}

    return parallel_tools


def build_worker_node(runner: WorkerRunnerProtocol):
    async def run_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
        call = WorkerCall.model_validate(payload["call"])
        dependencies = {
            key: CapabilityObservation.model_validate(value)
            for key, value in payload.get("dependency_observations", {}).items()
        }
        observation = await runner.run(
                call,
                teacher_id=payload.get("teacher_id"),
                class_id=payload.get("class_id"),
                dependency_observations=dependencies,
        )
        return {"pending_observations": [observation]}

    return run_worker


def decision_feedback(state: GraphStateInput) -> Dict[str, Any]:
    current = _state(state)
    assert current.validation_feedback is not None
    return {"pending_observations": [current.validation_feedback]}


def merge_observations(state: GraphStateInput) -> Dict[str, Any]:
    current = _state(state)
    new_items = current.pending_observations[current.merged_observation_count :]
    merged = dict(current.observations)
    for observation in new_items:
        merged[observation.result_key] = observation

    repeated = dict(current.repeated_call_counts)
    if current.decision is not None:
        for call in current.decision.current_calls:
            signature = call.signature()
            repeated[signature] = repeated.get(signature, 0) + 1

    tool_increment = len(current.decision.tool_calls) if current.decision else 0
    worker_increment = (
        1 if current.decision and current.decision.worker_calls else 0
    )
    new_citations = _citations_from_observations(new_items, current.citations)
    return {
        "observations": merged,
        "merged_observation_count": len(current.pending_observations),
        "react_step": current.react_step + 1,
        "tool_call_count": current.tool_call_count + tool_increment,
        "worker_batch_count": current.worker_batch_count + worker_increment,
        "repeated_call_counts": repeated,
        "citations": new_citations,
        "trace": [
            TraceEvent(
                step="merge_observations",
                message="Merged capability observations into Main state.",
                metadata={
                    "observations": [
                        {
                            "tool_name": item.capability_name,
                            "success": item.status is ObservationStatus.COMPLETED,
                        }
                        for item in new_items
                    ]
                },
            )
        ],
    }


def _citations_from_observations(
    observations: List[CapabilityObservation],
    existing: List[Citation],
) -> List[Citation]:
    """从 RAG evidence 中提取 API 可展示引用，不复制证据正文。"""

    known = {
        (item.source, item.title, item.section, item.page, item.url)
        for item in existing
    }
    found: List[Citation] = []

    def visit(value):
        if isinstance(value, dict):
            citation = value.get("citation")
            if isinstance(citation, dict):
                item = Citation(
                    source=str(
                        citation.get("source_id")
                        or citation.get("source")
                        or "retrieved_source"
                    ),
                    title=citation.get("title"),
                    section=citation.get("section"),
                    page=citation.get("page"),
                    url=citation.get("uri") or citation.get("url"),
                )
                identity = (
                    item.source,
                    item.title,
                    item.section,
                    item.page,
                    item.url,
                )
                if identity not in known:
                    known.add(identity)
                    found.append(item)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for observation in observations:
        visit(observation.data)
    return found


def finalize_draft(state: GraphStateInput) -> Dict[str, Any]:
    current = _state(state)
    assert current.decision is not None and current.decision.final_answer
    return {
        "workflow_status": WorkflowStatus.COMPLETED,
        "draft": Draft(
            title="EduFlow draft",
            content=current.decision.final_answer,
            is_draft=True,
        ),
        "approval": Approval(),
        "trace": [
            TraceEvent(
                step="finalize_draft",
                message="Main produced the teacher-facing draft.",
            )
        ],
    }


def clarification(state: GraphStateInput) -> Dict[str, Any]:
    current = _state(state)
    assert current.decision is not None and current.decision.clarification_question
    question = current.decision.clarification_question
    return {
        "needs_clarification": True,
        "clarification_question": question,
        "workflow_status": WorkflowStatus.COMPLETED,
        "draft": Draft(title="Clarification", content=question, is_draft=False),
        "trace": [
            TraceEvent(
                step="clarification",
                message="Main requested one clarification before continuing.",
            )
        ],
    }


def _bounded_fallback(state: GraphState) -> str:
    completed = [
        observation.capability_name
        for observation in state.observations.values()
        if observation.status is ObservationStatus.COMPLETED
    ]
    evidence = ", ".join(completed) if completed else "no verified capability results"
    return (
        "Draft: I could not complete every research step within the safe execution "
        f"limit. Available evidence came from: {evidence}. Please review this "
        "limitation or provide a narrower request."
    )


def build_main_react_graph(
    *,
    main_agent: Optional[MainAgentProtocol] = None,
    model_provider=None,
    registry: Optional[ToolRegistry] = None,
    worker_registry: Optional[WorkerRegistry] = None,
    worker_runner: Optional[WorkerRunnerProtocol] = None,
    context_manager: Optional[ContextManagerProtocol] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    long_memory_extractor: Optional[LongTermMemoryExtractorProtocol] = None,
    long_memory_store: Optional[LongTermMemoryStoreProtocol] = None,
    max_steps: int = 8,
    max_tool_calls: int = 12,
    max_worker_batches: int = 2,
    max_workers_per_batch: int = 3,
):
    resolved_store = long_memory_store or _default_store()
    resolved_registry = registry or build_default_tool_registry(resolved_store)
    resolved_provider = model_provider or ChatCompletionsModelProvider()
    resolved_workers = worker_registry or WorkerRegistry(DEFAULT_WORKER_PROFILES)
    resolved_agent = main_agent or MainReActAgent(resolved_provider)
    resolved_runner = worker_runner or BoundedWorkerRunner(
        provider=resolved_provider,
        tool_registry=resolved_registry,
        worker_registry=resolved_workers,
    )
    resolved_context = context_manager or ContextManager(
        long_term_memory_reader=resolved_store
    )
    resolved_extractor = long_memory_extractor or LLMLongTermMemoryExtractor()

    allowed_tools = {
        tool.name
        for tool in resolved_registry.list_tools()
        if tool.permission is ToolPermission.AUTO_EXECUTE
    }
    validator = MainDecisionValidator(
        resolved_registry,
        allowed_tool_names=allowed_tools,
        allowed_worker_names=resolved_workers.names,
    )
    tool_executor = MainToolExecutor(
        resolved_registry,
        allowed_tool_names=allowed_tools,
    )

    graph = StateGraph(GraphState)
    graph.add_node("initialize", initialize)
    graph.add_node(
        "main_react",
        build_main_react_node(
            resolved_agent,
            resolved_registry,
            resolved_workers,
            resolved_context,
            max_steps=max_steps,
        ),
    )
    graph.add_node(
        "validate_decision",
        build_validate_decision_node(
            validator,
            max_tool_calls=max_tool_calls,
            max_worker_batches=max_worker_batches,
            max_workers_per_batch=max_workers_per_batch,
        ),
    )
    graph.add_node("single_tool", build_single_tool_node(tool_executor))
    graph.add_node("parallel_tools", build_parallel_tools_node(tool_executor))
    graph.add_node("run_worker", build_worker_node(resolved_runner))
    graph.add_node("decision_feedback", decision_feedback)
    graph.add_node("merge_observations", merge_observations)
    graph.add_node("finalize_draft", finalize_draft)
    graph.add_node("clarification", clarification)
    graph.add_node("context_update", build_context_update_node(resolved_context))
    graph.add_node(
        "long_memory_update",
        build_long_memory_update_node(resolved_extractor, resolved_store),
    )

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "main_react")
    graph.add_edge("main_react", "validate_decision")
    graph.add_conditional_edges(
        "validate_decision",
        route_validated_decision,
        {
            ExecutionRoute.SINGLE_TOOL.value: "single_tool",
            ExecutionRoute.PARALLEL_TOOLS.value: "parallel_tools",
            ExecutionRoute.FEEDBACK.value: "decision_feedback",
            ExecutionRoute.CLARIFICATION.value: "clarification",
            ExecutionRoute.FINAL.value: "finalize_draft",
        },
    )
    graph.add_edge("single_tool", "merge_observations")
    graph.add_edge("parallel_tools", "merge_observations")
    graph.add_edge("run_worker", "merge_observations")
    graph.add_edge("decision_feedback", "merge_observations")
    graph.add_edge("merge_observations", "main_react")
    graph.add_edge("finalize_draft", "context_update")
    graph.add_edge("clarification", "context_update")
    graph.add_edge("context_update", "long_memory_update")
    graph.add_edge("long_memory_update", END)
    return graph.compile(checkpointer=checkpointer)


def _default_store() -> AsyncEduFlowStore:
    from app.config import settings

    return AsyncEduFlowStore(settings.database_url)
