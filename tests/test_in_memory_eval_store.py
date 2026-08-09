
"""Behavior tests for the deterministic, non-production evaluation store."""

from app.schemas import (
    LongTermMemoryAction,
    LongTermMemoryCandidate,
    LongTermMemoryOperation,
    LongTermMemoryScope,
    LongTermMemoryType,
    MemoryRetrievalMode,
)
from evals.in_memory_store import InMemoryEvalStore


def test_store_persists_and_reads_long_term_memory(tmp_path) -> None:
    store = InMemoryEvalStore()
    saved = store.save_long_term_memory(
        LongTermMemoryCandidate(
            scope=LongTermMemoryScope.TEACHER,
            scope_id="teacher-001",
            memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
            content="Prefers concise activity-plan steps.",
            reason="The teacher explicitly requested short steps more than once.",
        )
    )

    memories = store.list_long_term_memories(
        scope=LongTermMemoryScope.TEACHER.value,
        scope_id="teacher-001",
    )

    assert saved["memory_id"]
    assert len(memories) == 1
    assert memories[0]["content"] == "Prefers concise activity-plan steps."


def test_store_keeps_long_term_memories_scoped_to_their_owner(tmp_path) -> None:
    store = InMemoryEvalStore()
    store.save_long_term_memory(
        LongTermMemoryCandidate(
            scope=LongTermMemoryScope.TEACHER,
            scope_id="teacher-001",
            memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
            content="Prefers concise activity-plan steps.",
            reason="The teacher explicitly requested short steps more than once.",
        )
    )

    memories = store.list_long_term_memories(
        scope=LongTermMemoryScope.TEACHER.value,
        scope_id="teacher-002",
    )

    assert memories == []


def test_store_applies_insert_update_and_delete_operations(tmp_path) -> None:
    store = InMemoryEvalStore()
    initial = LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id="teacher-001",
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content="Prefers concise activity-plan steps.",
        reason="The teacher explicitly requested short steps.",
    )
    inserted = store.apply_long_term_memory_operation(
        LongTermMemoryOperation(
            action=LongTermMemoryAction.INSERT,
            candidate=initial,
            reason="A durable preference was explicitly stated.",
        ),
        teacher_id="teacher-001",
        class_id=None,
    )
    updated = store.apply_long_term_memory_operation(
        LongTermMemoryOperation(
            action=LongTermMemoryAction.UPDATE,
            memory_id=inserted["memory_id"],
            candidate=initial.model_copy(
                update={"content": "Prefers detailed activity-plan steps."}
            ),
            reason="The teacher explicitly changed the preference.",
        ),
        teacher_id="teacher-001",
        class_id=None,
    )

    deleted = store.apply_long_term_memory_operation(
        LongTermMemoryOperation(
            action=LongTermMemoryAction.DELETE,
            memory_id=inserted["memory_id"],
            reason="The teacher withdrew the preference.",
        ),
        teacher_id="teacher-001",
        class_id=None,
    )

    assert inserted["action"] == "insert"
    assert updated["content"] == "Prefers detailed activity-plan steps."
    assert deleted == {"action": "delete", "memory_id": inserted["memory_id"]}
    assert store.list_long_term_memories(scope="teacher", scope_id="teacher-001") == []


def test_store_rejects_an_update_that_moves_memory_to_a_different_owner(tmp_path) -> None:
    store = InMemoryEvalStore()
    saved = store.save_long_term_memory(
        LongTermMemoryCandidate(
            scope=LongTermMemoryScope.TEACHER,
            scope_id="teacher-001",
            memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
            content="Prefers concise activity-plan steps.",
            reason="Explicit preference.",
        )
    )

    import pytest

    with pytest.raises(ValueError, match="cannot change its owner"):
        store.apply_long_term_memory_operation(
            LongTermMemoryOperation(
                action=LongTermMemoryAction.UPDATE,
                memory_id=saved["memory_id"],
                candidate=LongTermMemoryCandidate(
                    scope=LongTermMemoryScope.TEACHER,
                    scope_id="teacher-002",
                    memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
                    content="Different teacher preference.",
                    reason="Invalid scope move.",
                ),
                reason="Attempt to change owner.",
            ),
            teacher_id="teacher-001",
            class_id=None,
        )


def test_store_loads_only_high_priority_active_teacher_profile_memories(tmp_path) -> None:
    store = InMemoryEvalStore()
    profile = LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id="teacher-001",
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content="Uses Australian English.",
        reason="Explicit preference.",
        retrieval_mode=MemoryRetrievalMode.PROFILE,
        importance=5,
    )
    store.save_long_term_memory(profile)
    store.save_long_term_memory(
        profile.model_copy(
            update={
                "content": "Prefers concise activity-plan steps.",
                "importance": 3,
            }
        )
    )
    store.save_long_term_memory(
        profile.model_copy(
            update={
                "content": "Previously used a seed-growing activity.",
                "retrieval_mode": MemoryRetrievalMode.RECALL_ONLY,
                "importance": 5,
            }
        )
    )

    profiles = store.list_profile_memories(teacher_id="teacher-001")

    assert [item["content"] for item in profiles] == [
        "Uses Australian English.",
        "Prefers concise activity-plan steps.",
    ]
