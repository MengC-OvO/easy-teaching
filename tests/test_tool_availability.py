from pydantic import BaseModel

from app.schemas import (
    CapabilityObservation,
    CapabilitySource,
    ObservationStatus,
    RiskLevel,
)
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolResult,
    available_tools_for_state,
)


class Empty(BaseModel):
    pass


def _tool(
    name: str,
    *,
    write: bool = False,
    max_successful_calls_per_run: int | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=name,
        category=ToolCategory.SYSTEM,
        input_model=Empty,
        output_model=Empty,
        risk_level=(RiskLevel.L2_CONTROLLED_WRITE if write else RiskLevel.L0_READ_ONLY),
        permission=(ToolPermission.REQUIRE_APPROVAL if write else ToolPermission.AUTO_EXECUTE),
        domain=ToolDomain.LOCAL,
        max_successful_calls_per_run=max_successful_calls_per_run,
        handler=lambda _: ToolResult.ok(
            data={},
            risk_level=(RiskLevel.L2_CONTROLLED_WRITE if write else RiskLevel.L0_READ_ONLY),
        ),
    )


def _names(**kwargs):
    tools = [
        _tool("get_class_context"),
        _tool("retrieve_knowledge"),
        _tool("query_records"),
        _tool("get_daily_context"),
        _tool("check_activity_safety", max_successful_calls_per_run=2),
        _tool("save_observation", write=True),
        _tool("export_records", write=True),
        _tool("drive_operation"),
    ]
    return {
        tool.name
        for tool in available_tools_for_state(
            tools,
            observations=kwargs.get("observations", {}),
            tool_attempt_counts=kwargs.get("tool_attempt_counts", {}),
        )
    }


def test_initial_state_exposes_all_registered_permitted_tools() -> None:
    names = _names()

    assert names == {
        "get_class_context",
        "retrieve_knowledge",
        "query_records",
        "get_daily_context",
        "check_activity_safety",
        "save_observation",
        "export_records",
        "drive_operation",
    }


def test_completed_one_shot_tool_and_exhausted_retrieval_are_hidden() -> None:
    observation = CapabilityObservation(
        result_key="class",
        capability_name="get_class_context",
        source_kind=CapabilitySource.TOOL,
        status=ObservationStatus.COMPLETED,
        data={"age_group": "3-5"},
    )

    names = _names(
        observations={"class": observation},
        tool_attempt_counts={"query_records": 2},
    )

    assert "get_class_context" not in names
    assert "query_records" not in names
    assert "retrieve_knowledge" in names


def test_safety_tool_allows_one_recheck_then_reaches_request_limit() -> None:
    first = CapabilityObservation(
        result_key="safety",
        capability_name="check_activity_safety",
        source_kind=CapabilitySource.TOOL,
        status=ObservationStatus.COMPLETED,
        data={"issues": [{"code": "water"}, {"code": "outdoor"}]},
    )

    names = _names(observations={"safety": first})

    assert "check_activity_safety" in names

    second = first.model_copy(
        update={
            "result_key": "safety_revision",
            "data": {"issues": [{"code": "outdoor"}]},
        }
    )
    names = _names(observations={"safety": first, "safety_revision": second})

    assert "check_activity_safety" not in names
