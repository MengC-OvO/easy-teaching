import asyncio
from datetime import datetime

from app.schemas import RiskLevel
from app.tools import ToolErrorCode, ToolExecutionContext, ToolRegistry
from app.tools.controlled_tools.export_records import build_export_records_tool
from app.tools.controlled_tools.records import (
    build_query_records_tool,
    build_save_observation_tool,
)


class StubRecordStore:
    def __init__(self):
        self.saved_observations = []
        self.exports = []
        self.last_query = None

    def query_records(self, **values):
        assert values["teacher_id"] == "teacher-1"
        assert values["class_id"] == "kangaroo-room"
        self.last_query = values
        return [self._observation()]

    def save_observation(self, **values):
        self.saved_observations.append(values)
        return self._observation(
            objective_text=values["objective_text"],
            idempotency_key=values["idempotency_key"],
        )

    def get_exportable_records(self, **values):
        assert values["record_ids"] == ["observation-1"]
        return [self._observation()]

    def save_record_export(self, **values):
        self.exports.append(values)
        return {
            "export_id": "export-1",
            "format": values["format"],
            "template_name": values["template_name"],
            "storage_path": values["storage_path"],
            "checksum": values["checksum"],
            "status": "ready",
            "expires_at": None,
        }

    @staticmethod
    def _observation(**overrides):
        result = {
            "record_type": "observation",
            "observation_id": "observation-1",
            "centre_id": "demo-centre",
            "class_id": "kangaroo-room",
            "author_teacher_id": "teacher-1",
            "child_ids": ["child-1"],
            "observed_at": "2026-08-25T09:00:00",
            "setting": "Outdoor area",
            "objective_text": "The child stacked four blocks and invited a peer to join.",
            "educator_actions": "The educator supplied more blocks.",
            "status": "draft",
            "version": 1,
            "created_at": "2026-08-25T10:00:00",
            "updated_at": "2026-08-25T10:00:00",
        }
        result.update({key: value for key, value in overrides.items() if key in result})
        return result


SCOPE = ToolExecutionContext(teacher_id="teacher-1", class_id="kangaroo-room")


def test_query_records_uses_only_trusted_execution_scope() -> None:
    store = StubRecordStore()
    registry = ToolRegistry()
    registry.register(build_query_records_tool(store))

    result = asyncio.run(
        registry.execute_async(
            "query_records",
            {
                "record_type": "observation",
                "search_text": "stacked four blocks",
                "limit": 5,
            },
            execution_context=SCOPE,
        )
    )

    assert result.success is True
    assert result.data["returned_count"] == 1
    assert result.data["search_text"] == "stacked four blocks"
    assert store.last_query["search_text"] == "stacked four blocks"


def test_save_observation_requires_approval_then_uses_reviewed_fields() -> None:
    store = StubRecordStore()
    registry = ToolRegistry()
    registry.register(build_save_observation_tool(store))
    arguments = {
        "child_ids": ["child-1"],
        "observed_at": datetime(2026, 8, 25, 9, 0),
        "setting": "Outdoor area",
        "objective_text": "The child stacked four blocks and invited a peer to join.",
        "educator_actions": "The educator supplied more blocks.",
        "status": "draft",
        "source_request_id": "request-1",
        "idempotency_key": "observation-request-1",
    }

    blocked = asyncio.run(
        registry.execute_async("save_observation", arguments, execution_context=SCOPE)
    )
    saved = asyncio.run(
        registry.execute_async(
            "save_observation",
            arguments,
            approved=True,
            execution_context=SCOPE,
        )
    )

    assert blocked.success is False
    assert blocked.error.code is ToolErrorCode.PERMISSION_DENIED
    assert saved.success is True
    assert saved.risk_level is RiskLevel.L2_CONTROLLED_WRITE
    assert store.saved_observations[0]["teacher_id"] == "teacher-1"
    assert store.saved_observations[0]["class_id"] == "kangaroo-room"


def test_export_records_creates_a_fixed_docx_from_saved_records(tmp_path) -> None:
    store = StubRecordStore()
    registry = ToolRegistry()
    registry.register(build_export_records_tool(store, export_root=tmp_path))

    result = asyncio.run(
        registry.execute_async(
            "export_records",
            {
                "record_ids": ["observation-1"],
                "format": "docx",
                "template_name": "observation",
            },
            approved=True,
            execution_context=SCOPE,
        )
    )

    assert result.success is True
    exported_path = tmp_path / result.data["storage_path"].split("\\")[-1]
    assert exported_path.exists()
    assert exported_path.stat().st_size > 0
    assert len(result.data["checksum"]) == 64
