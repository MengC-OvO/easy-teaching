from pydantic import ValidationError

from app.schemas import (
    LongTermMemoryCandidate,
    LongTermMemoryScope,
    LongTermMemoryType,
    MemoryRetrievalMode,
)


def test_long_term_memory_candidate_allows_teacher_preferences() -> None:
    candidate = LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id="teacher-001",
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content="Prefers concise activity-plan steps.",
        reason="The teacher explicitly requested short steps more than once.",
    )

    assert candidate.scope is LongTermMemoryScope.TEACHER
    assert candidate.memory_type is LongTermMemoryType.TEACHER_PREFERENCE


def test_long_term_memory_candidate_allows_confirmed_class_facts() -> None:
    candidate = LongTermMemoryCandidate(
        scope=LongTermMemoryScope.CLASS,
        scope_id="kangaroo-room",
        memory_type=LongTermMemoryType.CLASS_FACT,
        content="The class uses synthetic class data only.",
        reason="This is a confirmed class-level operating constraint.",
    )

    assert candidate.scope is LongTermMemoryScope.CLASS


def test_long_term_memory_candidate_rejects_wrong_scope_for_memory_type() -> None:
    try:
        LongTermMemoryCandidate(
            scope=LongTermMemoryScope.CLASS,
            scope_id="kangaroo-room",
            memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
            content="Prefers concise steps.",
            reason="Explicit teacher preference.",
        )
    except ValidationError as error:
        assert "teacher_preference memory must use teacher scope" in str(error)
    else:
        raise AssertionError("Teacher preferences must not be stored under a class")


def test_profile_memory_must_be_a_teacher_preference() -> None:
    try:
        LongTermMemoryCandidate(
            scope=LongTermMemoryScope.CLASS,
            scope_id="kangaroo-room",
            memory_type=LongTermMemoryType.CLASS_FACT,
            content="The room prefers outdoor play.",
            reason="Synthetic example.",
            retrieval_mode=MemoryRetrievalMode.PROFILE,
        )
    except ValidationError as error:
        assert "profile memory" in str(error)
    else:
        raise AssertionError("Only teacher preferences may be profile memory")
