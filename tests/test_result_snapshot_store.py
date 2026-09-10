"""SQL persistence/scope tests; SQLite does not prove PostgreSQL lock behavior."""
import asyncio

import pytest
from sqlalchemy.orm import Session

from tests.test_outbox_leases import store
from app.services.models import ConversationSessionRecord, ToolResultSnapshot
from app.services.observation_results import fingerprint


def test_snapshot_persistence_retry_scope_and_hash(store):
    with Session(store.test_engine) as db:
        db.add(ConversationSessionRecord(session_id="session", thread_id="thread",
            teacher_id="teacher", class_id="class"))
        db.commit()

    async def scenario():
        body = {"content": "immutable original"}
        row = dict(body_ref="a" * 64, request_id="req", session_id="session",
            teacher_id="teacher", class_id="class", result_key="result",
            content_hash=fingerprint(body), body=body)
        assert await store.save_tool_result_snapshot(**row) == "a" * 64
        assert await store.save_tool_result_snapshot(**row) == "a" * 64
        original = await store.read_tool_result_snapshot(body_ref=row["body_ref"],
            session_id="session", teacher_id="teacher", class_id="class")
        assert original["body"] == body
        with pytest.raises(ValueError, match="hash"):
            await store.save_tool_result_snapshot(**{**row, "body": {"content": "changed"}})
        with pytest.raises(ValueError, match="identity"):
            await store.save_tool_result_snapshot(**{**row, "result_key": "other"})
        with pytest.raises(PermissionError):
            await store.read_tool_result_snapshot(body_ref=row["body_ref"],
                session_id="session", teacher_id="intruder", class_id="class")
    asyncio.run(scenario())
    with Session(store.test_engine) as db:
        assert db.query(ToolResultSnapshot).count() == 1


def test_record_cursor_advances_across_both_tables_at_same_timestamp(store):
    from datetime import datetime
    from unittest.mock import AsyncMock
    from app.services.models import ObservationRecord, EducationalRecord
    from app.tools.controlled_tools.records import build_query_records_tool, QueryRecordsInput
    from app.tools.definition import ToolExecutionContext
    # Isolate SQL ordering here; permission enforcement has separate tests.
    store._require_class_access = AsyncMock()
    stamp = datetime(2026, 9, 1)
    with Session(store.test_engine) as db:
        for index in range(3):
            common = dict(centre_id="centre", class_id="class", author_teacher_id="teacher",
                          created_at=stamp, idempotency_key=f"key-{index}")
            db.add(ObservationRecord(observation_id=f"o{index}", observed_at=stamp,
                setting="room", objective_text="sample", **common))
            db.add(EducationalRecord(record_id=f"e{index}", record_type="reflection",
                title="sample", analysis="sample", **common))
        db.commit()

    async def scenario():
        tool = build_query_records_tool(store)
        context = ToolExecutionContext(teacher_id="teacher", class_id="class")
        cursor, found = None, []
        for _ in range(3):
            result = await tool.async_runtime_handler(QueryRecordsInput(limit=2, cursor=cursor), context)
            found.extend(row.get("observation_id") or row["record_id"] for row in result.data["records"])
            cursor = result.data["next_cursor"]
        assert found == ["o2", "o1", "o0", "e2", "e1", "e0"]
        assert cursor is None and result.data["has_more"] is False
        with pytest.raises(ValueError, match="cursor"):
            await store.query_records(teacher_id="teacher", class_id="class", record_type="all", cursor="invalid")
    asyncio.run(scenario())
