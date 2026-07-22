from app.schemas import (
    ContextBudget,
    ConversationMemory,
    ConversationRole,
    ConversationTurn,
    Draft,
    GraphState,
    Intent,
    ThreadContext,
    TraceEvent,
    WorkflowStatus,
)
from app.services import ContextManager


class StubMemoryUpdater:
    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory
        self.calls = []

    def update_memory(
        self,
        *,
        previous_memory,
        current_turns,
        archived_turns,
        max_summary_chars,
    ):
        self.calls.append(
            {
                "previous_memory": previous_memory,
                "current_turns": current_turns,
                "archived_turns": archived_turns,
                "max_summary_chars": max_summary_chars,
            }
        )
        return self.memory


class StubLongTermMemoryReader:
    def list_profile_memories(self, *, teacher_id, limit=4):
        assert teacher_id == "teacher-001"
        assert limit == 4
        return [
            {
                "memory_id": "memory-001",
                "scope": "teacher",
                "scope_id": "teacher-001",
                "memory_type": "teacher_preference",
                "content": "Prefers concise activity-plan steps.",
                "reason": "Explicit preference.",
            }
        ]


def test_context_manager_keeps_memory_unchanged_while_recent_turns_fit_budget() -> None:
    memory = ConversationMemory(
        conversation_goal="Complete an outdoor sensory activity draft.",
        important_requirements=["Use simple materials."],
        open_tasks=["Ask teacher for approval."],
        compact_summary="Outdoor sensory activity remains a draft.",
    )
    updater = StubMemoryUpdater(memory)
    manager = ContextManager(memory_updater=updater)
    state = GraphState(
        request_id="req-context",
        session_id="session-context",
        user_message="Plan an outdoor sensory walk.",
        intent=Intent.ACTIVITY_PLANNING,
        workflow_status=WorkflowStatus.COMPLETED,
        draft=Draft(title="Outdoor sensory walk", content="Draft plan."),
        trace=[TraceEvent(step="planning_react", message="Completed planning.")],
    )

    context = manager.update_after_run(state)

    assert context.thread_id == "session-context"
    assert [turn.role.value for turn in context.recent_turns] == ["user", "assistant"]
    assert context.memory == ConversationMemory()
    assert updater.calls == []
    assert context.tool_trace_summary[-1].step == "planning_react"


def test_context_manager_archives_old_turns_then_updates_memory() -> None:
    memory = ConversationMemory(
        conversation_goal="Finish the activity plan.",
        open_tasks=["Add more detail."],
        compact_summary="An activity plan is being revised.",
    )
    updater = StubMemoryUpdater(memory)
    manager = ContextManager(memory_updater=updater)
    state = GraphState(
        request_id="req-context-compress",
        session_id="session-context",
        user_message="Continue the plan.",
        workflow_status=WorkflowStatus.WAITING_FOR_APPROVAL,
        context=ThreadContext(budget=ContextBudget(max_recent_turns=2)),
    )

    first_context = manager.update_after_run(state)
    second_context = manager.update_after_run(
        state.model_copy(
            update={
                "request_id": "req-context-compress-2",
                "user_message": "Add more details.",
                "context": first_context,
            }
        )
    )

    assert [turn.content for turn in second_context.recent_turns] == [
        "Add more details.",
        "Waiting for teacher approval.",
    ]
    assert len(updater.calls) == 1
    assert [turn.content for turn in updater.calls[0]["archived_turns"]] == [
        "Continue the plan.",
        "Waiting for teacher approval.",
    ]
    assert second_context.memory == memory


def test_context_manager_builds_explicit_model_context_from_memory() -> None:
    manager = ContextManager(memory_updater=StubMemoryUpdater(ConversationMemory()))
    context = ThreadContext(
        memory=ConversationMemory(
            conversation_goal="Revise the activity plan.",
            important_requirements=["Use simple materials."],
            confirmed_preferences=["Teacher prefers short steps."],
            open_tasks=["Make it shorter."],
            compact_summary="An indoor activity draft is in progress.",
        ),
        recent_turns=[
            ConversationTurn(role=ConversationRole.USER, content="Make it shorter."),
        ],
    )

    prompt_context = manager.build_model_context(context)

    assert "Conversation goal" in prompt_context
    assert "Important requirements" in prompt_context
    assert "Open tasks" in prompt_context
    assert "Make it shorter." in prompt_context


def test_context_manager_adds_scoped_long_term_memory_to_model_context() -> None:
    manager = ContextManager(
        memory_updater=StubMemoryUpdater(ConversationMemory()),
        long_term_memory_reader=StubLongTermMemoryReader(),
    )

    prompt_context = manager.build_model_context(
        ThreadContext(),
        teacher_id="teacher-001",
    )

    assert "Teacher profile preferences" in prompt_context
    assert "Prefers concise activity-plan steps." in prompt_context


def test_context_manager_archives_turns_when_recent_token_budget_is_exceeded() -> None:
    updater = StubMemoryUpdater(ConversationMemory(compact_summary="Memory updated."))
    manager = ContextManager(memory_updater=updater)
    state = GraphState(
        request_id="req-token-budget",
        session_id="session-token-budget",
        user_message="x" * 120,
        context=ThreadContext(
            recent_turns=[
                ConversationTurn(role=ConversationRole.USER, content="y" * 120),
                ConversationTurn(role=ConversationRole.ASSISTANT, content="z" * 120),
            ],
            budget=ContextBudget(max_recent_turns=4, max_recent_tokens=30),
        ),
    )

    context = manager.update_after_run(state)

    assert len(context.recent_turns) == 2
    assert len(updater.calls[0]["archived_turns"]) == 2
