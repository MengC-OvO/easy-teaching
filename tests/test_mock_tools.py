from app.schemas import RiskLevel
from app.services import EduFlowStore
from app.tools import ToolErrorCode, build_mock_tool_registry


def make_store(tmp_path) -> EduFlowStore:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'eduflow-test.sqlite3'}")
    store.initialize()
    return store


def test_mock_tool_registry_registers_day2_tools(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    assert [tool.name for tool in registry.list_tools()] == [
        "get_class_profile",
        "search_policy_index",
        "save_draft",
    ]


def test_get_class_profile_tool_reads_synthetic_data(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    result = registry.execute("get_class_profile", {"class_id": "kangaroo-room"})

    assert result.success is True
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.data["name"] == "Kangaroo Room"
    assert "synthetic data only" in result.data["safety_notes"]
    assert result.trace is not None
    assert result.trace.tool_name == "get_class_profile"


def test_search_policy_index_tool_returns_synthetic_matches(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    result = registry.execute("search_policy_index", {"query": "program"})

    assert result.success is True
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.data["results"][0]["policy_id"] == "nqs-qa1-program"


def test_save_draft_tool_requires_approval(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert result.risk_level is RiskLevel.L2_CONTROLLED_WRITE


def test_save_draft_tool_writes_after_approval(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
        approved=True,
    )

    assert result.success is True
    assert result.data == {
        "draft_id": "draft-001",
        "draft_type": "activity_plan",
        "title": "Outdoor sensory walk",
        "status": "draft",
    }


def test_save_draft_tool_requires_idempotency_key(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
        approved=True,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    assert result.error.details["errors"][0]["loc"] == ("idempotency_key",)


def test_save_draft_tool_reuses_existing_result_for_same_idempotency_key(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    first = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
        approved=True,
    )
    second = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-002",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Should not replace original",
            "content": "Different synthetic content.",
        },
        approved=True,
    )

    assert first.success is True
    assert second.success is True
    assert second.data == first.data


def test_get_class_profile_tool_reports_missing_profile(tmp_path) -> None:
    registry = build_mock_tool_registry(make_store(tmp_path))

    result = registry.execute("get_class_profile", {"class_id": "missing-room"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_ERROR
    assert result.error.details == {"class_id": "missing-room"}
