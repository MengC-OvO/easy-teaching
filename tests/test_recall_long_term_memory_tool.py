import asyncio

from app.schemas import (
    LongTermMemoryCandidate,
    LongTermMemoryScope,
    LongTermMemoryType,
    MemoryRetrievalMode,
    CapabilityCall,
)
from app.services import EduFlowStore
from app.agents import MainToolExecutor
from app.tools import ToolExecutionContext, build_default_tool_registry


def _memory(*, owner: str, content: str, mode: MemoryRetrievalMode):
    return LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id=owner,
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content=content,
        reason="Synthetic teacher preference.",
        retrieval_mode=mode,
        importance=3,
    )


def test_recall_tool_returns_only_active_on_demand_memories_for_current_teacher(tmp_path) -> None:
    store = EduFlowStore(database_url=f"sqlite:///{tmp_path / 'eduflow.sqlite3'}")
    store.initialize()
    store.save_long_term_memory(
        _memory(
            owner="teacher-001",
            content="Previously used seed-growing activities outdoors.",
            mode=MemoryRetrievalMode.RECALL_ONLY,
        )
    )
    store.save_long_term_memory(
        _memory(
            owner="teacher-001",
            content="Uses Australian English.",
            mode=MemoryRetrievalMode.PROFILE,
        )
    )
    store.save_long_term_memory(
        _memory(
            owner="teacher-002",
            content="Previously used seed-growing activities indoors.",
            mode=MemoryRetrievalMode.RECALL_ONLY,
        )
    )
    registry = build_default_tool_registry(store)

    result = registry.execute(
        "recall_long_term_memory",
        {"query": "seed activities"},
        execution_context=ToolExecutionContext(teacher_id="teacher-001"),
    )

    assert result.success is True
    assert result.data["memories"] == [
        {
            "memory_id": result.data["memories"][0]["memory_id"],
            "memory_type": "teacher_preference",
            "content": "Previously used seed-growing activities outdoors.",
            "importance": 3,
        }
    ]


def test_react_executor_supplies_trusted_owner_scope_to_recall_tool(tmp_path) -> None:
    store = EduFlowStore(database_url=f"sqlite:///{tmp_path / 'eduflow.sqlite3'}")
    store.initialize()
    store.save_long_term_memory(
        _memory(
            owner="teacher-001",
            content="Previously used water-play activities outdoors.",
            mode=MemoryRetrievalMode.RECALL_ONLY,
        )
    )
    executor = MainToolExecutor(
        build_default_tool_registry(store),
        allowed_tool_names={"recall_long_term_memory"},
    )
    result = asyncio.run(
        executor.execute_one(
            CapabilityCall(
                name="recall_long_term_memory",
                arguments={"query": "water-play"},
                result_key="memory",
            ),
            teacher_id="teacher-001",
            class_id=None,
        )
    )

    assert result.status.value == "completed"
    assert result.data["memories"][0]["content"].startswith(
        "Previously used water-play"
    )
