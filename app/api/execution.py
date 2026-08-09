"""Background execution for accepted EduFlow API messages."""

from enum import Enum
from typing import Any, Dict, Optional, Set

from app.api.runtime import ApiRuntime
from app.schemas import GraphState, RunStatus, StreamEventType, TraceEvent, WorkflowStatus
from app.workflows import checkpoint_config


async def execute_message(
    *,
    runtime: ApiRuntime,
    request_id: str,
    session_id: str,
    thread_id: str,
    teacher_id: Optional[str] = None,
    class_id: Optional[str] = None,
    message: str,
) -> None:
    """Run one accepted message and persist its API-facing lifecycle status."""
    await runtime.store.update_conversation_run_status(request_id, RunStatus.RUNNING.value)
    state: Optional[GraphState] = None
    try:
        initial_state_model = GraphState(
            request_id=request_id,
            session_id=session_id,
            thread_id=thread_id,
            teacher_id=teacher_id,
            class_id=class_id,
            user_message=message,
        )
        # 只提交本次消息字段。未提交的 context 由同 thread checkpoint 保留；
        # initialize 节点会重置本轮临时 ReAct 字段。
        incremental_state = initial_state_model.model_dump(
            mode="json",
            exclude_defaults=True,
            exclude_none=True,
        )
        result = await runtime.graph.ainvoke(
            incremental_state,
            config=checkpoint_config(thread_id),
        )
        state = GraphState.model_validate(result)
        final_status = _run_status(state.workflow_status)
    except Exception:
        state = await _state_from_checkpoint(runtime, thread_id)
        final_status = RunStatus.FAILED
    await persist_run_outcome(
        runtime=runtime,
        request_id=request_id,
        session_id=session_id,
        state=state,
        final_status=final_status,
    )


async def execute_checkpoint_resume(
    *,
    runtime: ApiRuntime,
    request_id: str,
    session_id: str,
    thread_id: str,
) -> None:
    """Continue a non-interrupted run from its latest durable checkpoint."""
    state: Optional[GraphState] = None
    try:
        result = await runtime.graph.ainvoke(
            None,
            config=checkpoint_config(thread_id),
        )
        state = GraphState.model_validate(result)
        final_status = _run_status(state.workflow_status)
    except Exception:
        state = await _state_from_checkpoint(runtime, thread_id)
        final_status = RunStatus.FAILED
    await persist_run_outcome(
        runtime=runtime,
        request_id=request_id,
        session_id=session_id,
        state=state,
        final_status=final_status,
    )


async def persist_run_outcome(
    *,
    runtime: ApiRuntime,
    request_id: str,
    session_id: str,
    state: Optional[GraphState],
    final_status: RunStatus,
) -> None:
    """Persist one terminal or paused state and publish replay-safe events."""
    if state is not None:
        await _save_public_result(runtime, state)
    await runtime.store.update_conversation_run_status(request_id, final_status.value)
    if state is not None:
        await _publish_state_events(runtime, state, final_status)
        return
    await _append_event_once(
        runtime,
        request_id=request_id,
        session_id=session_id,
        event=StreamEventType.FAILED,
        data={"status": final_status.value},
    )


async def _save_public_result(runtime: ApiRuntime, state: GraphState) -> None:
    if state.draft is None:
        return
    await runtime.store.save_conversation_run_result(
        request_id=state.request_id,
        session_id=state.session_id,
        draft=state.draft.model_dump(mode="json"),
        approval=state.approval.model_dump(mode="json"),
        citations=[
            citation.model_dump(mode="json")
            for citation in state.citations[state.run_citation_start :]
        ],
    )


async def _state_from_checkpoint(
    runtime: ApiRuntime,
    thread_id: str,
) -> Optional[GraphState]:
    try:
        snapshot = await runtime.graph.aget_state(checkpoint_config(thread_id))
        if not snapshot.values:
            return None
        return GraphState.model_validate(snapshot.values)
    except Exception:
        return None


async def _publish_state_events(
    runtime: ApiRuntime,
    state: GraphState,
    final_status: RunStatus,
) -> None:
    await _publish_graph_trace(runtime, state)

    if state.draft is not None:
        await _append_event_once(
            runtime,
            request_id=state.request_id,
            session_id=state.session_id,
            event=StreamEventType.DRAFT_READY,
            data={
                "status": final_status.value,
                "title": state.draft.title,
            },
        )

    if final_status is RunStatus.WAITING_FOR_APPROVAL:
        await _append_event_once(
            runtime,
            request_id=state.request_id,
            session_id=state.session_id,
            event=StreamEventType.APPROVAL_REQUIRED,
            data={
                "status": final_status.value,
                "risk_level": state.approval.risk_level.value,
                "reason": state.approval.reason,
            },
        )
    elif final_status is RunStatus.COMPLETED:
        await _append_event_once(
            runtime,
            request_id=state.request_id,
            session_id=state.session_id,
            event=StreamEventType.COMPLETED,
            data={"status": final_status.value},
        )
    elif final_status is RunStatus.FAILED:
        await _append_event_once(
            runtime,
            request_id=state.request_id,
            session_id=state.session_id,
            event=StreamEventType.FAILED,
            data={"status": final_status.value},
        )


async def _publish_graph_trace(runtime: ApiRuntime, state: GraphState) -> None:
    existing_indexes: Set[int] = {
        event["data"]["trace_index"]
        for event in await runtime.store.list_conversation_events(
            request_id=state.request_id,
        )
        if event["event"] == StreamEventType.TRACE.value
        and event["data"].get("origin") == "graph"
        and isinstance(event["data"].get("trace_index"), int)
    }
    for trace_index, trace in enumerate(
        state.trace[state.run_trace_start :],
        start=state.run_trace_start,
    ):
        if trace_index in existing_indexes:
            continue
        await _append_event(
            runtime,
            request_id=state.request_id,
            session_id=state.session_id,
            event=StreamEventType.TRACE,
            data={
                "origin": "graph",
                "trace_index": trace_index,
                "step": trace.step,
                "message": trace.message,
                "metadata": _safe_trace_metadata(trace),
            },
        )


_SAFE_TRACE_METADATA_KEYS = {
    "actions",
    "applied_operations",
    "attempts",
    "citation_count",
    "code",
    "confidence",
    "created",
    "current_step",
    "decision",
    "evidence_count",
    "fallback",
    "generation_error_code",
    "generation_fallback",
    "implementation",
    "intent",
    "interrupt_enabled",
    "loaded_skill_name",
    "max_attempts",
    "memory_ids",
    "needs_clarification",
    "open_tasks",
    "record_id",
    "recoverable",
    "recent_turns",
    "redacted_types",
    "replacement_count",
    "retry_exhausted",
    "request_id",
    "risk_level",
    "status",
    "status_code",
    "stop_reason",
    "structured_attempts",
    "structured_max_attempts",
    "summary_chars",
    "thread_id",
}


def _safe_trace_metadata(trace: TraceEvent) -> Dict[str, Any]:
    safe = {
        key: _safe_json_value(value)
        for key, value in trace.metadata.items()
        if key in _SAFE_TRACE_METADATA_KEYS
    }
    retrieval = trace.metadata.get("retrieval")
    if isinstance(retrieval, dict):
        safe["retrieval"] = {
            key: _safe_json_value(value)
            for key, value in retrieval.items()
            if key
            in {
                "requested_top_k",
                "mode",
                "reranker",
                "raw_result_count",
                "dense_result_count",
                "bm25_result_count",
                "deduplicated_count",
                "returned_count",
                "reranked",
            }
        }
    observations = trace.metadata.get("observations")
    if isinstance(observations, list):
        safe["observations"] = [
            {
                key: _safe_json_value(value)
                for key, value in observation.items()
                if key in {"tool_name", "success", "error_code"}
            }
            for observation in observations
            if isinstance(observation, dict)
        ]
    return safe


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    return str(value)


async def _append_event(
    runtime: ApiRuntime,
    *,
    request_id: str,
    session_id: str,
    event: StreamEventType,
    data: dict,
) -> None:
    await runtime.store.append_conversation_event(
        request_id=request_id,
        session_id=session_id,
        event=event.value,
        data=data,
    )


async def _append_event_once(
    runtime: ApiRuntime,
    *,
    request_id: str,
    session_id: str,
    event: StreamEventType,
    data: dict,
) -> None:
    existing = await runtime.store.list_conversation_events(request_id=request_id)
    if any(item["event"] == event.value and item["data"] == data for item in existing):
        return
    await _append_event(
        runtime,
        request_id=request_id,
        session_id=session_id,
        event=event,
        data=data,
    )


def _run_status(workflow_status: WorkflowStatus) -> RunStatus:
    if workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL:
        return RunStatus.WAITING_FOR_APPROVAL
    if workflow_status is WorkflowStatus.COMPLETED:
        return RunStatus.COMPLETED
    if workflow_status is WorkflowStatus.FAILED:
        return RunStatus.FAILED
    return RunStatus.RUNNING
