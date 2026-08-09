"""Shared state, context, and memory nodes for the production Main ReAct graph."""

import inspect
from typing import Any, Dict, List, Mapping, Optional, Protocol, Union

from app.schemas import (
    ConversationTurn,
    GraphState,
    LongTermMemoryOperation,
    ThreadContext,
    TraceEvent,
)
from app.services import ModelProviderError


GraphStateInput = Union[GraphState, Mapping[str, Any]]


class ContextManagerProtocol(Protocol):
    async def build_model_context_async(
        self,
        context: ThreadContext,
        *,
        teacher_id: Optional[str] = None,
    ) -> str:
        ...

    async def update_after_run_async(self, state: GraphState) -> ThreadContext:
        ...


class LongTermMemoryExtractorProtocol(Protocol):
    async def decide_async(
        self,
        *,
        turns: List[ConversationTurn],
        existing_memories: List[Dict[str, str]],
        teacher_id: Optional[str] = None,
        class_id: Optional[str] = None,
    ) -> List[LongTermMemoryOperation]:
        ...


class LongTermMemoryStoreProtocol(Protocol):
    async def list_memories_for_owners(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        limit: int = 12,
    ) -> List[Dict[str, str]]:
        ...

    async def apply_long_term_memory_operation(
        self,
        operation: LongTermMemoryOperation,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> Dict[str, str]:
        ...


def _state(state: GraphStateInput) -> GraphState:
    return state if isinstance(state, GraphState) else GraphState.model_validate(state)


async def _resolve(value):
    return await value if inspect.isawaitable(value) else value


def initialize(state: GraphStateInput) -> Dict[str, Any]:
    current = _state(state)
    thread_id = current.thread_id or current.context.thread_id or current.session_id
    return {
        "thread_id": thread_id,
        "context": current.context.model_copy(update={"thread_id": thread_id}),
        "decision": None,
        "execution_route": None,
        "validation_feedback": None,
        "observations": {},
        "merged_observation_count": len(current.pending_observations),
        "react_step": 0,
        "tool_call_count": 0,
        "worker_batch_count": 0,
        "repeated_call_counts": {},
        "run_trace_start": len(current.trace),
        "run_citation_start": len(current.citations),
        "trace": [
            TraceEvent(
                step="initialize",
                message="Initialized EduFlow graph state.",
                metadata={"thread_id": thread_id},
            )
        ],
    }


def build_context_update_node(context_manager: ContextManagerProtocol):
    async def context_update(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        current_run = current.model_copy(
            update={"trace": current.trace[current.run_trace_start :]}
        )
        updater = getattr(context_manager, "update_after_run_async", None)
        context = await _resolve(
            updater(current_run)
            if updater is not None
            else context_manager.update_after_run(current_run)
        )
        return {
            "context": context,
            "trace": [
                TraceEvent(
                    step="context_update",
                    message="Updated compressed thread context.",
                    metadata={
                        "thread_id": context.thread_id,
                        "recent_turns": len(context.recent_turns),
                        "open_tasks": len(context.memory.open_tasks),
                        "summary_chars": len(context.memory.compact_summary),
                    },
                )
            ],
        }

    return context_update


def build_long_memory_update_node(
    extractor: LongTermMemoryExtractorProtocol,
    store: LongTermMemoryStoreProtocol,
):
    async def long_memory_update(state: GraphStateInput) -> Dict[str, Any]:
        current = _state(state)
        if current.teacher_id is None and current.class_id is None:
            return {"trace": [_memory_trace(0)]}
        try:
            existing = await _resolve(
                store.list_memories_for_owners(
                    teacher_id=current.teacher_id,
                    class_id=current.class_id,
                )
            )
            decide = getattr(extractor, "decide_async", None)
            operations = await _resolve(
                decide(
                    turns=current.context.recent_turns[-2:],
                    existing_memories=existing,
                    teacher_id=current.teacher_id,
                    class_id=current.class_id,
                )
                if decide is not None
                else extractor.decide(
                    turns=current.context.recent_turns[-2:],
                    existing_memories=existing,
                    teacher_id=current.teacher_id,
                    class_id=current.class_id,
                )
            )
            applied = []
            for operation in operations:
                applied.append(
                    await _resolve(
                        store.apply_long_term_memory_operation(
                            operation,
                            teacher_id=current.teacher_id,
                            class_id=current.class_id,
                        )
                    )
                )
        except (ModelProviderError, TypeError, ValueError) as error:
            return {
                "trace": [
                    TraceEvent(
                        step="long_memory_update",
                        message="Long-term memory update was skipped after an error.",
                        metadata={"applied_operations": 0, "error": str(error)},
                    )
                ]
            }
        return {
            "trace": [
                TraceEvent(
                    step="long_memory_update",
                    message="Applied long-term memory operations.",
                    metadata={
                        "applied_operations": len(applied),
                        "actions": [item["action"] for item in applied],
                        "memory_ids": [
                            item["memory_id"] for item in applied if "memory_id" in item
                        ],
                    },
                )
            ]
        }

    return long_memory_update


def _memory_trace(applied: int) -> TraceEvent:
    return TraceEvent(
        step="long_memory_update",
        message="Skipped long-term memory update without an owner.",
        metadata={"applied_operations": applied},
    )
