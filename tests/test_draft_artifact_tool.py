import asyncio

from app.schemas import RiskLevel
from app.tools import (
    ToolErrorCode,
    ToolExecutionContext,
    ToolRegistry,
    build_read_draft_artifact_tool,
)


class StubArtifactStore:
    def __init__(self) -> None:
        self.calls = []

    async def get_conversation_artifact(self, **scope):
        self.calls.append(scope)
        if scope["source_request_id"] == "missing":
            return None
        return {
            "source_request_id": scope["source_request_id"],
            "title": "Nature sensory activity",
            "content": "Full draft content " * 100,
            "content_chars": len("Full draft content " * 100),
            "created_at": "2026-08-25T09:00:00",
            "status": "unsaved",
        }


SCOPE = ToolExecutionContext(
    teacher_id="teacher-1",
    class_id="kangaroo-room",
    session_id="session-1",
)


def test_read_draft_artifact_returns_complete_scoped_content() -> None:
    store = StubArtifactStore()
    registry = ToolRegistry()
    registry.register(build_read_draft_artifact_tool(store))

    result = asyncio.run(
        registry.execute_async(
            "read_draft_artifact",
            {"source_request_id": "request-1"},
            execution_context=SCOPE,
        )
    )

    assert result.success is True
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.data["content"] == "Full draft content " * 100
    assert store.calls == [
        {
            "source_request_id": "request-1",
            "session_id": "session-1",
            "teacher_id": "teacher-1",
            "class_id": "kangaroo-room",
        }
    ]


def test_read_draft_artifact_requires_trusted_scope() -> None:
    registry = ToolRegistry()
    registry.register(build_read_draft_artifact_tool(StubArtifactStore()))

    result = asyncio.run(
        registry.execute_async(
            "read_draft_artifact",
            {"source_request_id": "request-1"},
            execution_context=ToolExecutionContext(session_id="session-1"),
        )
    )

    assert result.success is False
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED


def test_read_draft_artifact_does_not_leak_unavailable_draft() -> None:
    registry = ToolRegistry()
    registry.register(build_read_draft_artifact_tool(StubArtifactStore()))

    result = asyncio.run(
        registry.execute_async(
            "read_draft_artifact",
            {"source_request_id": "missing"},
            execution_context=SCOPE,
        )
    )

    assert result.success is False
    assert result.error.code is ToolErrorCode.EXECUTION_ERROR
