from app.schemas import (
    ConversationRole,
    ConversationTurn,
    GraphState,
    LongTermMemoryAction,
    LongTermMemoryCandidate,
    LongTermMemoryOperation,
    LongTermMemoryScope,
    LongTermMemoryType,
    ThreadContext,
)
from app.workflows.main_graph import build_long_memory_update_node


class StubLongTermMemoryExtractor:
    def __init__(self, operations):
        self.operations = operations
        self.calls = []

    def decide(self, *, turns, existing_memories, teacher_id=None, class_id=None):
        self.calls.append(
            {
                "turns": turns,
                "existing_memories": existing_memories,
                "teacher_id": teacher_id,
                "class_id": class_id,
            }
        )
        return self.operations


class StubLongTermMemoryStore:
    def __init__(self, existing=None):
        self.existing = existing or []
        self.applied = []

    def list_memories_for_owners(self, *, teacher_id, class_id, limit=12):
        return self.existing

    def apply_long_term_memory_operation(self, operation, *, teacher_id, class_id):
        self.applied.append(operation)
        return {"action": operation.action.value, "memory_id": "memory-1"}


def test_long_memory_update_applies_operations_from_the_latest_exchange() -> None:
    candidate = LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id="teacher-001",
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content="Prefers concise activity-plan steps.",
        reason="The teacher explicitly requested short steps twice.",
    )
    operation = LongTermMemoryOperation(
        action=LongTermMemoryAction.INSERT,
        candidate=candidate,
        reason="A durable preference was explicitly stated.",
    )
    extractor = StubLongTermMemoryExtractor([operation])
    store = StubLongTermMemoryStore(existing=[{"memory_id": "old-memory"}])
    node = build_long_memory_update_node(extractor, store)

    result = node(
        GraphState(
            request_id="req-memory",
            session_id="session-memory",
            teacher_id="teacher-001",
            user_message="Keep it concise.",
            context=ThreadContext(
                recent_turns=[
                    ConversationTurn(role=ConversationRole.USER, content="Older message."),
                    ConversationTurn(role=ConversationRole.ASSISTANT, content="Older reply."),
                    ConversationTurn(role=ConversationRole.USER, content="Keep it concise."),
                    ConversationTurn(role=ConversationRole.ASSISTANT, content="Done."),
                ]
            ),
        )
    )

    assert [turn.content for turn in extractor.calls[0]["turns"]] == ["Keep it concise.", "Done."]
    assert extractor.calls[0]["existing_memories"] == [{"memory_id": "old-memory"}]
    assert store.applied == [operation]
    assert result["trace"][0].metadata["actions"] == ["insert"]


def test_long_memory_update_skips_extraction_without_an_owner() -> None:
    extractor = StubLongTermMemoryExtractor([])
    store = StubLongTermMemoryStore()
    node = build_long_memory_update_node(extractor, store)

    result = node(
        GraphState(
            request_id="req-memory-skip",
            session_id="session-memory-skip",
            user_message="Keep it concise.",
        )
    )

    assert extractor.calls == []
    assert store.applied == []
    assert result["trace"][0].metadata["applied_operations"] == 0
