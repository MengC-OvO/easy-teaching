from app.schemas import RiskLevel, SpecialistKind, get_specialist_permission
from app.skills import SkillLoader, build_default_skill_registry
from app.tools import ToolErrorCode, ToolRegistry, build_load_skill_tool


REGISTERED_PLANNING_TOOLS = {
    "load_skill",
    "get_class_profile",
    "align_to_eylf_outcomes",
    "retrieve_risk_guidance",
    "check_activity_safety",
    "recall_long_term_memory",
}


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        build_load_skill_tool(
            loader=SkillLoader(build_default_skill_registry()),
            permission=get_specialist_permission(SpecialistKind.PLANNING),
            registered_tool_names=REGISTERED_PLANNING_TOOLS,
        )
    )
    return registry


def test_load_skill_tool_returns_registered_skill_content() -> None:
    result = build_registry().execute(
        "load_skill",
        {"skill_name": "activity_planning"},
        allowed_tool_names={"load_skill"},
    )

    assert result.success is True
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.data["manifest"]["name"] == "activity_planning"
    assert result.data["manifest"]["specialist"] == "planning"
    assert "# Activity Planning Skill" in result.data["instructions"]
    assert len(result.data["content_hash"]) == 64


def test_load_skill_tool_returns_structured_error_for_unknown_skill() -> None:
    result = build_registry().execute(
        "load_skill",
        {"skill_name": "unknown_skill"},
        allowed_tool_names={"load_skill"},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    assert result.error.recoverable is True
