import json

import pytest

from app.schemas import (
    SpecialistKind,
    SpecialistPermissionDenied,
    SpecialistPermissionPolicy,
    get_specialist_permission,
)
from app.skills import (
    DuplicateSkillError,
    SkillLoadError,
    SkillLoader,
    SkillNotFoundError,
    SkillRegistry,
    SkillRegistryError,
    build_default_skill_registry,
)


REGISTERED_PLANNING_TOOLS = {
    "load_skill",
    "get_class_profile",
    "align_to_eylf_outcomes",
    "retrieve_risk_guidance",
    "check_activity_safety",
    "recall_long_term_memory",
}


def write_skill(tmp_path, *, name="test_skill", tools=None):
    skill_directory = tmp_path / name
    skill_directory.mkdir()
    (skill_directory / "SKILL.md").write_text(
        "# Test Skill\n\nFollow the trusted test workflow.",
        encoding="utf-8",
    )
    (skill_directory / "manifest.json").write_text(
        json.dumps(
            {
                "name": name,
                "version": "1.0",
                "specialist": "planning",
                "description": "Synthetic test Skill.",
                "required_tool_names": tools or ["get_class_profile"],
                "optional_tool_names": [],
                "output_model": "ActivityPlan",
            }
        ),
        encoding="utf-8",
    )
    return skill_directory


def test_default_loader_reads_file_based_planning_skill() -> None:
    loader = SkillLoader(build_default_skill_registry())

    loaded = loader.load(
        "activity_planning",
        permission=get_specialist_permission(SpecialistKind.PLANNING),
        registered_tool_names=REGISTERED_PLANNING_TOOLS,
    )

    assert loaded.manifest.name == "activity_planning"
    assert loaded.manifest.output_model == "ActivityPlan"
    assert "# Activity Planning Skill" in loaded.instructions
    assert len(loaded.content_hash) == 64


def test_registry_resolves_only_registered_names(tmp_path) -> None:
    write_skill(tmp_path)
    registry = SkillRegistry(tmp_path)
    registry.register("test_skill", "test_skill")

    assert registry.resolve("test_skill") == (tmp_path / "test_skill").resolve()
    with pytest.raises(SkillNotFoundError):
        registry.resolve("unknown_skill")


def test_registry_rejects_duplicate_and_escaping_directories(tmp_path) -> None:
    write_skill(tmp_path)
    registry = SkillRegistry(tmp_path)
    registry.register("test_skill", "test_skill")

    with pytest.raises(DuplicateSkillError):
        registry.register("test_skill", "test_skill")
    with pytest.raises(SkillRegistryError, match="escapes"):
        registry.register("escape", "../")


def test_loader_rejects_skill_tool_outside_specialist_permission(tmp_path) -> None:
    write_skill(tmp_path, tools=["send_family_message"])
    registry = SkillRegistry(tmp_path)
    registry.register("test_skill", "test_skill")

    with pytest.raises(SpecialistPermissionDenied, match="send_family_message"):
        SkillLoader(registry).load(
            "test_skill",
            permission=get_specialist_permission(SpecialistKind.PLANNING),
            registered_tool_names={"send_family_message"},
        )


def test_loader_rejects_unregistered_tool_even_when_policy_allows_it(tmp_path) -> None:
    write_skill(tmp_path, tools=["developer_added_tool"])
    registry = SkillRegistry(tmp_path)
    registry.register("test_skill", "test_skill")
    permission = SpecialistPermissionPolicy(
        specialist=SpecialistKind.PLANNING,
        allowed_tool_names=frozenset({"developer_added_tool"}),
        max_steps=7,
    )

    with pytest.raises(SkillLoadError, match="unregistered tools"):
        SkillLoader(registry).load(
            "test_skill",
            permission=permission,
            registered_tool_names=set(),
        )


def test_loader_rejects_manifest_name_mismatch(tmp_path) -> None:
    write_skill(tmp_path, name="manifest_name")
    registry = SkillRegistry(tmp_path)
    registry.register("registry_name", "manifest_name")

    with pytest.raises(SkillLoadError, match="does not match"):
        SkillLoader(registry).load(
            "registry_name",
            permission=get_specialist_permission(SpecialistKind.PLANNING),
            registered_tool_names=REGISTERED_PLANNING_TOOLS,
        )
