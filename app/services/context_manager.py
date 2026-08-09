import inspect
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from app.schemas import (
    ConversationMemory,
    ConversationRole,
    ConversationTurn,
    Draft,
    GraphError,
    GraphState,
    ThreadContext,
    TraceEvent,
    WorkflowStatus,
)
from app.services.context_summarizer import LLMContextSummarizer
from app.services.model_errors import ModelProviderError


class ConversationMemoryUpdater(Protocol):
    def update_memory(
        self,
        *,
        previous_memory: ConversationMemory,
        current_turns: List[ConversationTurn],
        archived_turns: List[ConversationTurn],
        max_summary_chars: int,
    ) -> ConversationMemory:
        ...


class LongTermMemoryReader(Protocol):
    def list_profile_memories(
        self,
        *,
        teacher_id: Optional[str],
        limit: int = 4,
    ) -> List[Dict[str, str]]:
        ...


class ContextManager:
    """Owns thread-level prompt memory, not the operational graph state."""

    def __init__(
        self,
        memory_updater: Optional[ConversationMemoryUpdater] = None,
        long_term_memory_reader: Optional[LongTermMemoryReader] = None,
    ) -> None:
        self.memory_updater = memory_updater or LLMContextSummarizer()
        self.long_term_memory_reader = long_term_memory_reader

    def update_after_run(self, state: GraphState) -> ThreadContext:
        thread_id = state.thread_id or state.context.thread_id or state.session_id
        current_turns = [self._user_turn(state), self._assistant_turn(state)]
        archived_turns, recent_turns = self._partition_turns(
            [*state.context.recent_turns, *current_turns],
            max_recent_turns=state.context.budget.max_recent_turns,
            max_recent_tokens=state.context.budget.max_recent_tokens,
        )
        # Recent turns are retained verbatim. Only ask an LLM to update
        # semantic memory when older turns must leave that bounded window.
        memory = state.context.memory
        if archived_turns:
            memory = self._update_memory(
                previous_memory=memory,
                current_turns=current_turns,
                archived_turns=archived_turns,
                max_summary_chars=state.context.budget.max_memory_summary_chars,
            )
        return ThreadContext(
            thread_id=thread_id,
            recent_turns=recent_turns,
            memory=memory,
            tool_trace_summary=[*state.context.tool_trace_summary, *state.trace],
            budget=state.context.budget,
        )

    def build_model_context(
        self,
        context: ThreadContext,
        *,
        teacher_id: Optional[str] = None,
    ) -> str:
        """Project thread memory into the bounded prompt context for an LLM."""
        memory = context.memory
        blocks = []
        if memory.conversation_goal:
            blocks.append(f"Conversation goal:\n{memory.conversation_goal}")
        if memory.important_requirements:
            blocks.append(self._list_block("Important requirements", memory.important_requirements))
        if memory.confirmed_preferences:
            blocks.append(self._list_block("Confirmed preferences", memory.confirmed_preferences))
        if memory.completed_work:
            blocks.append(self._list_block("Completed work", memory.completed_work))
        if memory.open_tasks:
            blocks.append(self._list_block("Open tasks", memory.open_tasks))
        if memory.compact_summary:
            blocks.append(f"Compact memory summary:\n{memory.compact_summary}")
        if context.recent_turns:
            blocks.append(
                "Recent conversation:\n"
                + "\n".join(
                    f"{turn.role.value}: {turn.content}"
                    for turn in context.recent_turns
                )
            )
        profile_memories = self._load_profile_memories(teacher_id)
        if profile_memories:
            blocks.append(
                "Teacher profile preferences:\n"
                + "\n".join(
                    f"- [{memory['memory_type']}] {memory['content']}"
                    for memory in profile_memories
                )
            )
        return "\n\n".join(blocks)

    async def build_model_context_async(
        self,
        context: ThreadContext,
        *,
        teacher_id: Optional[str] = None,
    ) -> str:
        memory = context.memory
        blocks = []
        if memory.conversation_goal:
            blocks.append(f"Conversation goal:\n{memory.conversation_goal}")
        if memory.important_requirements:
            blocks.append(self._list_block("Important requirements", memory.important_requirements))
        if memory.confirmed_preferences:
            blocks.append(self._list_block("Confirmed preferences", memory.confirmed_preferences))
        if memory.completed_work:
            blocks.append(self._list_block("Completed work", memory.completed_work))
        if memory.open_tasks:
            blocks.append(self._list_block("Open tasks", memory.open_tasks))
        if memory.compact_summary:
            blocks.append(f"Compact memory summary:\n{memory.compact_summary}")
        if context.recent_turns:
            blocks.append(
                "Recent conversation:\n"
                + "\n".join(
                    f"{turn.role.value}: {turn.content}" for turn in context.recent_turns
                )
            )
        profile_memories = []
        if self.long_term_memory_reader is not None:
            profile_memories = self.long_term_memory_reader.list_profile_memories(
                teacher_id=teacher_id,
                limit=4,
            )
            if inspect.isawaitable(profile_memories):
                profile_memories = await profile_memories
        if profile_memories:
            blocks.append(
                "Teacher profile preferences:\n"
                + "\n".join(
                    f"- [{item['memory_type']}] {item['content']}"
                    for item in profile_memories
                )
            )
        return "\n\n".join(blocks)

    async def update_after_run_async(self, state: GraphState) -> ThreadContext:
        thread_id = state.thread_id or state.context.thread_id or state.session_id
        current_turns = [self._user_turn(state), self._assistant_turn(state)]
        archived_turns, recent_turns = self._partition_turns(
            [*state.context.recent_turns, *current_turns],
            max_recent_turns=state.context.budget.max_recent_turns,
            max_recent_tokens=state.context.budget.max_recent_tokens,
        )
        memory = state.context.memory
        if archived_turns:
            updater = getattr(self.memory_updater, "update_memory_async", None)
            if updater is None:
                memory = self._update_memory(
                    previous_memory=memory,
                    current_turns=current_turns,
                    archived_turns=archived_turns,
                    max_summary_chars=state.context.budget.max_memory_summary_chars,
                )
            else:
                try:
                    memory = await updater(
                        previous_memory=memory,
                        current_turns=current_turns,
                        archived_turns=archived_turns,
                        max_summary_chars=state.context.budget.max_memory_summary_chars,
                    )
                except (ModelProviderError, TypeError, ValueError):
                    memory = self._fallback_memory(
                        memory,
                        current_turns,
                        state.context.budget.max_memory_summary_chars,
                    )
        return ThreadContext(
            thread_id=thread_id,
            recent_turns=recent_turns,
            memory=memory,
            tool_trace_summary=[*state.context.tool_trace_summary, *state.trace],
            budget=state.context.budget,
        )

    def _load_profile_memories(
        self,
        teacher_id: Optional[str],
    ) -> List[Dict[str, str]]:
        if self.long_term_memory_reader is None:
            return []
        return self.long_term_memory_reader.list_profile_memories(
            teacher_id=teacher_id,
            limit=4,
        )

    def _partition_turns(
        self,
        turns: Sequence[ConversationTurn],
        *,
        max_recent_turns: int,
        max_recent_tokens: int,
    ) -> Tuple[List[ConversationTurn], List[ConversationTurn]]:
        if max_recent_turns <= 0:
            return list(turns), []
        overflow_count = max(0, len(turns) - max_recent_turns)
        archived_turns = list(turns[:overflow_count])
        recent_turns = list(turns[overflow_count:])
        # Keep the latest user/assistant exchange intact even if it alone is large.
        while len(recent_turns) > 2 and self._estimate_tokens(recent_turns) > max_recent_tokens:
            archived_turns.append(recent_turns.pop(0))
        return archived_turns, recent_turns

    def _update_memory(
        self,
        *,
        previous_memory: ConversationMemory,
        current_turns: List[ConversationTurn],
        archived_turns: List[ConversationTurn],
        max_summary_chars: int,
    ) -> ConversationMemory:
        try:
            return self.memory_updater.update_memory(
                previous_memory=previous_memory,
                current_turns=current_turns,
                archived_turns=archived_turns,
                max_summary_chars=max_summary_chars,
            )
        except (ModelProviderError, TypeError, ValueError):
            return self._fallback_memory(previous_memory, current_turns, max_summary_chars)

    def _fallback_memory(
        self,
        previous_memory: ConversationMemory,
        current_turns: List[ConversationTurn],
        max_summary_chars: int,
    ) -> ConversationMemory:
        latest_user_message = current_turns[0].content if current_turns else None
        summary = previous_memory.compact_summary
        if latest_user_message:
            summary = self._limit(
                f"{summary} Latest request: {latest_user_message}".strip(),
                max_summary_chars,
            )
        return previous_memory.model_copy(
            update={
                "conversation_goal": previous_memory.conversation_goal or latest_user_message,
                "compact_summary": summary,
            }
        )

    def _user_turn(self, state: GraphState) -> ConversationTurn:
        return ConversationTurn(
            role=ConversationRole.USER,
            content=state.user_message,
            intent=state.intent,
            workflow_status=state.workflow_status,
            metadata={"request_id": state.request_id},
        )

    def _assistant_turn(self, state: GraphState) -> ConversationTurn:
        return ConversationTurn(
            role=ConversationRole.ASSISTANT,
            content=self._assistant_content(state),
            intent=state.intent,
            workflow_status=state.workflow_status,
            metadata={
                "citations": len(state.citations),
                "errors": len(state.errors),
                "safety_flags": len(state.safety_flags),
            },
        )

    def _assistant_content(self, state: GraphState) -> str:
        if state.needs_clarification and state.clarification_question:
            return state.clarification_question
        if state.draft:
            return self._format_draft(state.draft)
        if state.errors:
            return self._format_errors(state.errors)
        if state.workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL:
            return state.approval.reason or "Waiting for teacher approval."
        return f"Workflow finished with status: {state.workflow_status.value}."

    def _format_draft(self, draft: Draft) -> str:
        title = f"{draft.title}: " if draft.title else ""
        return f"{title}{self._limit(draft.content, 800)}"

    def _format_errors(self, errors: Sequence[GraphError]) -> str:
        return "; ".join(
            f"{error.code}: {self._limit(error.message, 160)}" for error in errors
        )

    def _estimate_tokens(self, turns: Sequence[ConversationTurn]) -> int:
        return sum(max(1, len(turn.content) // 4) + 4 for turn in turns)

    def _list_block(self, label: str, values: List[str]) -> str:
        return label + ":\n" + "\n".join(f"- {value}" for value in values)

    def _limit(self, value: str, max_length: int) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= max_length:
            return normalized
        return normalized[: max_length - 3].rstrip() + "..."
