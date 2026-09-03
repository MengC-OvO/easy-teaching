"""EasyTeaching 生产用统一 Main ReAct LangGraph。"""

import json
import inspect

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
    ApprovalStatus,
    CapabilityCall,
    CapabilityObservation,
    Draft,
    Citation,
    GraphError,
    GraphState,
    MainDecision,
    ObservationStatus,
    RiskLevel,
    TraceEvent,
    WorkerCall,
    WorkflowStatus,
)
from app.services import (
    ChatCompletionsModelProvider,
    ContextManager,
    AsyncEasyTeachingStore,
    LLMLongTermMemoryExtractor,
    ModelProviderError,
    build_model_observation_view,
)
from app.services.request_guard import (
    EasyTeachingRequestGuard,
    RequestGuardAction,
    RequestGuardResult,
)
from app.tools import (
    ToolExecutionContext,
    ToolPermission,
    ToolRegistry,
    build_default_tool_registry,
    available_tools_for_state,
    resolve_required_controlled_tools,
)
from app.tools.controlled_tools.check_activity_safety import activity_content_fingerprint
from app.workflows.main_react_support import (
    ContextManagerProtocol,
    LongTermMemoryExtractorProtocol,
    LongTermMemoryStoreProtocol,
    build_context_update_node,
    build_long_memory_update_node,
    initialize,
)


GraphStateInput = Union[GraphState, Mapping[str, Any]]


def _has_matching_activity_safety_observation(
    final_answer: str,
    observations: Mapping[str, CapabilityObservation],
) -> bool:
    expected = activity_content_fingerprint(final_answer)
    return any(
        observation.capability_name == "check_activity_safety"
        and observation.status is ObservationStatus.COMPLETED
        and observation.data.get("content_fingerprint") == expected
        for observation in observations.values()
    )


def _loaded_draft_references(
    observations: Mapping[str, CapabilityObservation],
) -> Dict[str, Dict[str, Optional[str]]]:
    """Return every server-verified draft loaded in this run."""

    loaded: Dict[str, Dict[str, Optional[str]]] = {}
    for key, observation in observations.items():
        if (
            observation.capability_name != "read_draft_artifact"
            or observation.status is not ObservationStatus.COMPLETED
        ):
            continue
        source_request_id = observation.data.get("source_request_id")
        if isinstance(source_request_id, str) and source_request_id:
            loaded[key] = {
                "source_request_id": source_request_id,
                "title": observation.data.get("title"),
            }
    return loaded


def _latest_checked_activity(
    observations: Mapping[str, CapabilityObservation],
) -> Optional[str]:
    for observation in reversed(list(observations.values())):
        if (
            observation.capability_name == "check_activity_safety"
            and observation.status is ObservationStatus.COMPLETED
        ):
            content = observation.data.get("recovery_content")
            if isinstance(content, str) and content.strip():
                return content
    return None


class MainAgentProtocol(Protocol):
    async def decide(self, **kwargs) -> MainDecision:
        ...


class WorkerRunnerProtocol(Protocol):
    async def run(self, call: WorkerCall, **kwargs) -> CapabilityObservation:
        ...


class RequestGuardProtocol(Protocol):
    def evaluate(
        self,
        user_message: str,
        *,
        conversation_context: str = "",
    ) -> RequestGuardResult:
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
    request_guard: RequestGuardProtocol,
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

        tools = available_tools_for_state(
            registry.list_tools(),
            observations=current.observations,
            tool_attempt_counts=current.tool_attempt_counts,
        )
        required_completion_actions = list(
            dict.fromkeys(
                [
                    *current.required_completion_actions,
                    *resolve_required_controlled_tools(
                        current.user_message,
                        registry.list_tools(),
                    ),
                ]
            )
        )
        try:
            context_builder = getattr(context_manager, "build_model_context_async", None)
            conversation_context = (
                await context_builder(
                    current.context,
                    teacher_id=current.teacher_id,
                    class_id=current.class_id,
                    session_id=current.session_id,
                )
                if context_builder is not None
                else context_manager.build_model_context(
                    current.context,
                    teacher_id=current.teacher_id,
                )
            )
            guard_result = request_guard.evaluate(
                current.user_message,
                conversation_context=conversation_context,
            )
            if guard_result.action is not RequestGuardAction.ALLOW:
                decision = (
                    MainDecision(
                        reason="The request is outside the allowed professional boundary.",
                        final_answer=guard_result.response,
                    )
                    if guard_result.action is RequestGuardAction.BLOCK
                    else MainDecision(
                        reason="The request needs education-scope clarification.",
                        clarification_question=guard_result.response,
                    )
                )
                return {
                    "decision": decision,
                    "trace": [
                        TraceEvent(
                            step="request_guard",
                            message="Applied the professional scope and injection boundary.",
                            metadata={
                                "status": guard_result.action.value,
                                "code": guard_result.code,
                            },
                        )
                    ],
                }
            decision = await agent.decide(
                    user_message=current.user_message,
                    conversation_context=conversation_context,
                    observations=current.observations,
                    available_tools=tools,
                    available_workers=worker_registry.public_descriptions(),
                    current_step=current.react_step,
                    max_steps=max_steps,
                    required_completion_actions=required_completion_actions,
                    loaded_draft_references=_loaded_draft_references(
                        current.observations
                    ),
            )
            tool_schema_chars = len(
                json.dumps(
                    [tool.model_spec() for tool in tools],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            observation_view_chars = len(
                json.dumps(
                    build_model_observation_view(current.observations),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
            return {
                "decision": decision,
                "required_completion_actions": required_completion_actions,
                "available_tool_names": [tool.name for tool in tools],
                "trace": [
                    TraceEvent(
                        step="main_react",
                        message="Main selected the current executable action.",
                        metadata={
                            "current_step": current.react_step,
                            "available_tools": [tool.name for tool in tools],
                            "tool_schema_chars": tool_schema_chars,
                            "observation_view_chars": observation_view_chars,
                            "conversation_context_chars": len(conversation_context),
                        },
                    )
                ],
            }
        except (ModelProviderError, TypeError, ValueError) as error:
            safe_metadata = (
                error.safe_metadata()
                if isinstance(error, ModelProviderError)
                else {"code": "invalid_main_decision", "recoverable": True}
            )
            return {
                "decision": MainDecision(
                    reason="The model decision was unavailable.",
                    final_answer=_model_unavailable_fallback(current, safe_metadata),
                ),
                "errors": [
                    GraphError(
                        code="main_react_model_error",
                        message="Main ReAct used a safe provider-error fallback.",
                        recoverable=True,
                    )
                ],
                "trace": [
                    TraceEvent(
                        step="main_react",
                        message="Main ReAct stopped after a model provider error.",
                        metadata=safe_metadata,
                    )
                ],
            }

    return main_react


def build_validate_decision_node(
    validator: MainDecisionValidator,
    *,
    max_steps: int,
    max_tool_calls: int,
    max_worker_batches: int,
    max_workers_per_batch: int,
):
    def validate_decision(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        assert current.decision is not None
        decision = current.decision

        normalized_dependency_batch = False
        normalized_result_keys = {}
        used_result_keys = set(current.observations)

        def unique_call_result_key(call):
            original = call.result_key
            if original not in used_result_keys:
                used_result_keys.add(original)
                return call
            suffix = max(1, current.react_step + 1)
            candidate = f"{original}__step_{suffix}"
            while candidate in used_result_keys:
                suffix += 1
                candidate = f"{original}__step_{suffix}"
            used_result_keys.add(candidate)
            normalized_result_keys[original] = candidate
            return call.model_copy(update={"result_key": candidate})

        if decision.tool_calls:
            decision = decision.model_copy(
                update={
                    "tool_calls": [
                        unique_call_result_key(call) for call in decision.tool_calls
                    ]
                }
            )
        elif decision.worker_calls:
            decision = decision.model_copy(
                update={
                    "worker_calls": [
                        unique_call_result_key(call) for call in decision.worker_calls
                    ]
                }
            )

        if len(decision.tool_calls) > 1:
            read_calls = []
            write_calls = []
            for call in decision.tool_calls:
                tool = validator.registry.get(call.name)
                if tool is None:
                    continue
                permission = tool.permission_for(call.arguments)
                if permission is ToolPermission.REQUIRE_APPROVAL:
                    write_calls.append(call)
                elif permission is ToolPermission.AUTO_EXECUTE:
                    read_calls.append(call)
            if read_calls and write_calls:
                # A model may correctly identify both operations but put a dependent
                # read and write in one invalid batch. Execute one safe prerequisite,
                # then let Main reconsider with its observation; never auto-execute
                # or preserve the write portion of the mixed batch.
                decision = MainDecision(
                    task_type=decision.task_type,
                    requires_activity_safety=decision.requires_activity_safety,
                    reason=(
                        "Execute the first read-only prerequisite from a mixed "
                        "read/write batch before reconsidering the controlled write."
                    ),
                    tool_calls=[read_calls[0]],
                )
                normalized_dependency_batch = True
            elif any(
                validator.registry.get(call.name) is not None
                and not validator.registry.get(call.name).parallel_safe
                for call in decision.tool_calls
            ):
                # The model selected useful reads but batched a capability whose
                # implementation is intentionally sequential (for example deep
                # RAG). Preserve model choice and execute the first call only.
                decision = MainDecision(
                    task_type=decision.task_type,
                    requires_activity_safety=decision.requires_activity_safety,
                    reason=(
                        "Execute the first call from a batch containing a "
                        "non-parallel capability, then reconsider the remainder."
                    ),
                    tool_calls=[decision.tool_calls[0]],
                )
                normalized_dependency_batch = True

        feedback = None
        normalized_safety_final = False
        normalized_safety_recheck = False
        if (
            decision.final_answer
            and decision.requires_activity_safety
            and not _has_matching_activity_safety_observation(
                decision.final_answer,
                current.observations,
            )
            and "check_activity_safety" not in current.available_tool_names
        ):
            # The request-scoped safety budget is exhausted. Do not let another
            # model rewrite create an unchecked variant and do not churn until the
            # graph step fallback. Finalize the exact last inspected candidate.
            checked_activity = _latest_checked_activity(current.observations)
            if checked_activity:
                decision = decision.model_copy(
                    update={
                        "final_answer": checked_activity,
                        "reason": (
                            "Safety budget exhausted; finalized the exact latest "
                            "Tool-checked activity version."
                        ),
                    }
                )
                normalized_safety_final = True
        elif (
            decision.final_answer
            and decision.requires_activity_safety
            and not _has_matching_activity_safety_observation(
                decision.final_answer,
                current.observations,
            )
            and "check_activity_safety" in current.available_tool_names
        ):
            # Safety is an execution invariant, not a suggestion for Main to
            # remember on its next turn. Check the exact candidate that Main
            # proposed instead of feeding the same instruction back into the
            # model and risking an unproductive rewrite/recheck loop.
            decision = MainDecision(
                task_type=decision.task_type,
                requires_activity_safety=True,
                reason="Check the exact proposed final activity version.",
                tool_calls=[
                    CapabilityCall(
                        name="check_activity_safety",
                        arguments={"activity_text": decision.final_answer},
                        result_key=(
                            f"auto_safety_step_{current.react_step}_"
                            f"{len(current.observations)}"
                        ),
                    )
                ],
            )
            normalized_safety_recheck = True
        unavailable_calls = [
            call.name
            for call in decision.tool_calls
            if call.name not in current.available_tool_names
        ]
        if unavailable_calls:
            feedback = validator.feedback(
                "这些工具在当前执行状态下已不可用，请根据已有结果结束或选择"
                "仍可用的工具：" + ", ".join(unavailable_calls)
            )
        elif decision.final_answer and current.required_completion_actions:
            feedback = validator.feedback(
                "教师明确要求的受控操作尚未进入审批预览："
                + ", ".join(current.required_completion_actions)
                + "。继续使用已有证据准备相应受控工具调用；不要只返回草稿。"
            )
        elif (
            decision.final_answer
            and current.react_step < max_steps
            and validator.registry.get("check_activity_safety") is not None
            and decision.requires_activity_safety
            and not _has_matching_activity_safety_observation(
                decision.final_answer,
                current.observations,
            )
        ):
            feedback = validator.feedback(
                "活动或教学方案的最终完整文本必须与一次 "
                "check_activity_safety 的已检查版本完全一致；请检查当前最终版本。"
            )
        elif (
            decision.tool_calls
            and current.tool_call_count + len(decision.tool_calls) > max_tool_calls
        ):
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
            "decision": decision,
            "execution_route": validation.route.value,
            "validation_feedback": validation.feedback,
            "trace": [
                TraceEvent(
                    step="validate_decision",
                    message="Validated the current Main decision.",
                    metadata={
                        "status": validation.route.value,
                        "normalized_dependency_batch": normalized_dependency_batch,
                        "normalized_safety_final": normalized_safety_final,
                        "normalized_safety_recheck": normalized_safety_recheck,
                        "normalized_result_keys": normalized_result_keys,
                        "requested_tools": [
                            call.name for call in current.decision.tool_calls
                        ],
                        "requested_workers": [
                            call.name.value for call in current.decision.worker_calls
                        ],
                        "feedback": (
                            validation.feedback.error.get("message")
                            if validation.feedback and validation.feedback.error
                            else None
                        ),
                        "required_completion_actions": current.required_completion_actions,
                    },
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
                session_id=current.session_id,
                request_id=current.request_id,
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
                session_id=current.session_id,
                request_id=current.request_id,
        )
        return {"pending_observations": observations}

    return parallel_tools


def build_worker_node(runner: WorkerRunnerProtocol):
    async def run_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
        call = WorkerCall.model_validate(payload["call"])
        observation = await runner.run(
                call,
                teacher_id=payload.get("teacher_id"),
                class_id=payload.get("class_id"),
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
    attempts = dict(current.tool_attempt_counts)
    executed_tools = current.execution_route in {
        ExecutionRoute.SINGLE_TOOL.value,
        ExecutionRoute.PARALLEL_TOOLS.value,
    }
    executed_workers = current.execution_route == ExecutionRoute.PARALLEL_WORKERS.value
    if current.decision is not None and (executed_tools or executed_workers):
        for call in current.decision.current_calls:
            signature = call.signature()
            repeated[signature] = repeated.get(signature, 0) + 1
    if current.decision is not None and executed_tools:
        for call in current.decision.tool_calls:
            attempts[call.name] = attempts.get(call.name, 0) + 1

    tool_increment = (
        len(current.decision.tool_calls)
        if current.decision is not None and executed_tools
        else 0
    )
    worker_increment = (
        1
        if current.decision is not None
        and current.decision.worker_calls
        and executed_workers
        else 0
    )
    new_citations = _citations_from_observations(new_items, current.citations)
    return {
        "observations": merged,
        "merged_observation_count": len(current.pending_observations),
        "react_step": current.react_step + 1,
        "tool_call_count": current.tool_call_count + tool_increment,
        "worker_batch_count": current.worker_batch_count + worker_increment,
        "repeated_call_counts": repeated,
        "tool_attempt_counts": attempts,
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
                            "status": item.status.value,
                            "error_code": (
                                item.error.get("code") if item.error else None
                            ),
                            "contract": _observation_contract(item),
                        }
                        for item in new_items
                    ]
                },
            )
        ],
    }


def route_merged_observations(state: GraphStateInput) -> str:
    """Fail closed when the latest knowledge retrieval misses the score gate."""

    current = _state(state)
    latest_rag = next(
        (
            observation
            for observation in reversed(list(current.observations.values()))
            if observation.capability_name == "retrieve_knowledge"
        ),
        None,
    )
    if (
        latest_rag is not None
        and latest_rag.data.get("answerability") == "insufficient"
    ):
        return "evidence_refusal"
    return "main_react"


def finalize_evidence_refusal(state: GraphStateInput) -> Dict[str, Any]:
    """Return a deterministic refusal without another answer-model call."""

    current = _state(state)
    latest_rag = next(
        observation
        for observation in reversed(list(current.observations.values()))
        if observation.capability_name == "retrieve_knowledge"
    )
    reason = str(
        latest_rag.data.get("answerability_reason")
        or "evidence_below_relevance_threshold"
    )
    return {
        "workflow_status": WorkflowStatus.COMPLETED,
        "draft": Draft(
            title="Insufficient evidence",
            content=(
                "The current knowledge base does not contain sufficiently reliable "
                "evidence to answer this question."
            ),
            is_draft=False,
        ),
        "approval": Approval(),
        "trace": [
            TraceEvent(
                step="finalize_evidence_refusal",
                message="Stopped before answer generation because RAG evidence missed the calibrated gate.",
                metadata={"reason": reason},
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
            title=current.decision.artifact_title or "EasyTeaching draft",
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


def _observation_contract(
    observation: CapabilityObservation,
) -> Dict[str, Any]:
    """Non-sensitive execution facts used by traces and behavioural evals."""

    allowed = {
        "knowledge_scope",
        "strategy",
        "returned_count",
        "source_request_id",
        "status",
    }
    return {
        key: value
        for key, value in observation.data.items()
        if key in allowed and isinstance(value, (str, int, float, bool, type(None)))
    }


def build_prepare_approval_node(store, registry: ToolRegistry):
    async def prepare_approval(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        assert current.decision is not None and len(current.decision.tool_calls) == 1
        call = current.decision.tool_calls[0]
        tool = registry.get(call.name)
        assert tool is not None
        assert tool.permission_for(call.arguments) is ToolPermission.REQUIRE_APPROVAL
        if not current.teacher_id:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "errors": [
                    GraphError(
                        code="write_requires_teacher",
                        message="A controlled write requires an authenticated teacher.",
                        recoverable=True,
                    )
                ],
            }
        try:
            validated_input = tool.input_model.model_validate(call.arguments)
            arguments = validated_input.model_dump(mode="json")
            preview = arguments
            if tool.approval_preparation_handler is not None:
                prepared = tool.approval_preparation_handler(
                    validated_input,
                    ToolExecutionContext(
                        teacher_id=current.teacher_id,
                        class_id=current.class_id,
                        session_id=current.session_id,
                        request_id=current.request_id,
                    ),
                )
                if inspect.isawaitable(prepared):
                    prepared = await prepared
                arguments = tool.input_model.model_validate(
                    prepared.arguments
                ).model_dump(mode="json")
                preview = prepared.preview
        except (PermissionError, TypeError, ValueError) as error:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "draft": Draft(
                    title="Save unavailable",
                    content=(
                        "I could not resolve the referenced draft for a safe "
                        "approval preview. Please identify or regenerate the draft."
                    ),
                    is_draft=False,
                ),
                "errors": [
                    GraphError(
                        code="approval_preparation_failed",
                        message=str(error),
                        recoverable=True,
                    )
                ],
                "trace": [
                    TraceEvent(
                        step="prepare_approval",
                        message="Could not prepare the controlled-write preview.",
                        metadata={"tool_name": call.name},
                    )
                ],
            }
        action = await store.create_tool_action_request(
            request_id=current.request_id,
            session_id=current.session_id,
            teacher_id=current.teacher_id,
            class_id=current.class_id,
            tool_name=call.name,
            arguments=arguments,
            preview=preview,
        )
        return {
            "workflow_status": WorkflowStatus.WAITING_FOR_APPROVAL,
            "draft": Draft(
                title=f"Confirm {call.name}",
                content=json.dumps(arguments, ensure_ascii=False, indent=2),
                is_draft=True,
            ),
            "approval": Approval(
                status=ApprovalStatus.REQUIRED,
                risk_level=tool.risk_for(arguments),
                reason="Review the frozen fields before this write is executed.",
                action_id=action["action_id"],
                tool_name=call.name,
                preview=preview,
            ),
            "trace": [
                TraceEvent(
                    step="prepare_approval",
                    message="Prepared a frozen controlled-write action for teacher approval.",
                    metadata={"action_id": action["action_id"], "tool_name": call.name},
                )
            ],
        }

    return prepare_approval


def _bounded_fallback(state: GraphState) -> str:
    selected_draft = next(
        (
            observation.data.get("content")
            for observation in reversed(list(state.observations.values()))
            if observation.capability_name == "read_draft_artifact"
            and observation.status is ObservationStatus.COMPLETED
            and isinstance(observation.data.get("content"), str)
        ),
        None,
    )
    if selected_draft:
        return (
            str(selected_draft)
            + "\n\n[The requested revision could not be completed within the "
            "bounded execution limit; the selected original draft is preserved.]"
        )
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


def _model_unavailable_fallback(
    state: GraphState,
    metadata: Mapping[str, Any],
) -> str:
    recovery_content = next(
        (
            str(observation.data.get("recovery_content") or "").strip()
            for observation in reversed(list(state.observations.values()))
            if observation.status is ObservationStatus.COMPLETED
            and str(observation.data.get("recovery_content") or "").strip()
        ),
        "",
    )
    if recovery_content:
        return (
            recovery_content
            + "\n\n[Recovery note: final model formatting was unavailable. This is "
            "the exact last Tool-checked draft; review the reported safety issues "
            "before use.]"
        )
    worker_summaries = [
        (
            observation.capability_name,
            str(observation.data.get("summary") or "").strip(),
        )
        for observation in state.observations.values()
        if observation.source_kind.value == "worker"
        and observation.status in {
            ObservationStatus.COMPLETED,
            ObservationStatus.INSUFFICIENT,
        }
        and str(observation.data.get("summary") or "").strip()
    ]
    if worker_summaries:
        sections = "\n\n".join(
            f"### {name}\n{summary[:4_000]}" for name, summary in worker_summaries
        )
        return (
            "EasyTeaching preserved the completed research results, but the final "
            "synthesis model became unavailable. Review the evidence summaries below."
            "\n\n"
            + sections
        )
    completed = [
        observation.capability_name
        for observation in state.observations.values()
        if observation.status is ObservationStatus.COMPLETED
    ]
    evidence = ", ".join(completed) if completed else "no completed capability results"
    code = str(metadata.get("code") or "provider_error")
    return (
        "EasyTeaching could not generate the requested draft because the model provider "
        f"became unavailable ({code}). Completed results were preserved from: "
        f"{evidence}. Please retry after the provider recovers; restating the request "
        "is not required."
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
    request_guard: Optional[RequestGuardProtocol] = None,
    max_steps: int = 8,
    max_tool_calls: int = 12,
    max_worker_batches: int = 1,
    max_workers_per_batch: int = 2,
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
    resolved_request_guard = request_guard or EasyTeachingRequestGuard()

    allowed_tools = {
        tool.name
        for tool in resolved_registry.list_tools()
        if tool.permission is not ToolPermission.FORBIDDEN
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
            resolved_request_guard,
            max_steps=max_steps,
        ),
    )
    graph.add_node(
        "validate_decision",
        build_validate_decision_node(
            validator,
            max_steps=max_steps,
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
    graph.add_node("finalize_evidence_refusal", finalize_evidence_refusal)
    graph.add_node("finalize_draft", finalize_draft)
    graph.add_node("clarification", clarification)
    graph.add_node(
        "prepare_approval",
        build_prepare_approval_node(resolved_store, resolved_registry),
    )
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
            ExecutionRoute.APPROVAL.value: "prepare_approval",
            ExecutionRoute.FEEDBACK.value: "decision_feedback",
            ExecutionRoute.CLARIFICATION.value: "clarification",
            ExecutionRoute.FINAL.value: "finalize_draft",
        },
    )
    graph.add_edge("single_tool", "merge_observations")
    graph.add_edge("parallel_tools", "merge_observations")
    graph.add_edge("run_worker", "merge_observations")
    graph.add_edge("decision_feedback", "merge_observations")
    graph.add_conditional_edges(
        "merge_observations",
        route_merged_observations,
        {
            "main_react": "main_react",
            "evidence_refusal": "finalize_evidence_refusal",
        },
    )
    graph.add_edge("finalize_draft", "context_update")
    graph.add_edge("clarification", "context_update")
    graph.add_edge("prepare_approval", "context_update")
    graph.add_edge("finalize_evidence_refusal", "context_update")
    graph.add_edge("context_update", "long_memory_update")
    graph.add_edge("long_memory_update", END)
    return graph.compile(checkpointer=checkpointer)


def _default_store() -> AsyncEasyTeachingStore:
    from app.config import settings

    return AsyncEasyTeachingStore(settings.database_url)
