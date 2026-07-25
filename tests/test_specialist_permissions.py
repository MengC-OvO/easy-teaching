import pytest
from pydantic import ValidationError

from app.schemas import (
    DEFAULT_SPECIALIST_PERMISSIONS,
    ForbiddenSpecialistAction,
    SpecialistKind,
    SpecialistPermissionDenied,
    SpecialistPermissionPolicy,
    get_specialist_permission,
)


def test_default_permissions_cover_every_specialist() -> None:
    assert set(DEFAULT_SPECIALIST_PERMISSIONS) == set(SpecialistKind)
    for specialist, permission in DEFAULT_SPECIALIST_PERMISSIONS.items():
        assert permission.specialist is specialist


def test_planning_permission_matches_current_react_boundary() -> None:
    permission = get_specialist_permission(SpecialistKind.PLANNING)

    assert permission.max_steps == 7
    assert permission.allowed_tool_names == {
        "load_skill",
        "get_class_profile",
        "retrieve_risk_guidance",
        "check_activity_safety",
        "align_to_eylf_outcomes",
        "save_draft",
        "recall_long_term_memory",
    }


@pytest.mark.parametrize(
    "specialist",
    [
        SpecialistKind.DOCUMENTATION,
        SpecialistKind.POLICY,
        SpecialistKind.FAMILY,
    ],
)
def test_fixed_workflows_have_no_function_calling_tools(
    specialist: SpecialistKind,
) -> None:
    permission = get_specialist_permission(specialist)

    assert permission.max_steps == 1
    assert permission.allowed_tool_names == frozenset()


def test_every_specialist_forbids_high_risk_autonomous_actions() -> None:
    for permission in DEFAULT_SPECIALIST_PERMISSIONS.values():
        assert ForbiddenSpecialistAction.REAL_WORLD_SEND in permission.forbidden_actions
        assert ForbiddenSpecialistAction.RAW_PII_OUTPUT in permission.forbidden_actions
        assert ForbiddenSpecialistAction.CHILD_DIAGNOSIS in permission.forbidden_actions
        assert ForbiddenSpecialistAction.MEDICAL_ADVICE in permission.forbidden_actions
        assert (
            ForbiddenSpecialistAction.LEGAL_COMPLIANCE_CONCLUSION
            in permission.forbidden_actions
        )
        assert ForbiddenSpecialistAction.UNAPPROVED_WRITE in permission.forbidden_actions


def test_permission_policy_rejects_invalid_execution_budget() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        SpecialistPermissionPolicy(
            specialist=SpecialistKind.PLANNING,
            max_steps=0,
        )


def test_permission_policy_rejects_blank_tool_name() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        SpecialistPermissionPolicy(
            specialist=SpecialistKind.PLANNING,
            allowed_tool_names=frozenset({"  "}),
            max_steps=1,
        )


def test_default_permission_models_and_mapping_are_immutable() -> None:
    permission = get_specialist_permission(SpecialistKind.PLANNING)

    with pytest.raises(ValidationError):
        permission.max_steps = 99

    with pytest.raises(TypeError):
        DEFAULT_SPECIALIST_PERMISSIONS[SpecialistKind.PLANNING] = permission


def test_permission_policy_blocks_unlisted_tool() -> None:
    permission = get_specialist_permission(SpecialistKind.PLANNING)

    permission.require_tool("get_class_profile")

    with pytest.raises(SpecialistPermissionDenied, match="cannot use tool"):
        permission.require_tool("send_family_message")


def test_permission_policy_blocks_steps_outside_budget() -> None:
    permission = get_specialist_permission(SpecialistKind.PLANNING)

    permission.require_step(0)
    permission.require_step(6)

    with pytest.raises(SpecialistPermissionDenied, match="7-step budget"):
        permission.require_step(7)


def test_permission_policy_blocks_structured_forbidden_action() -> None:
    permission = get_specialist_permission(SpecialistKind.FAMILY)

    with pytest.raises(SpecialistPermissionDenied, match="real_world_send"):
        permission.require_action(ForbiddenSpecialistAction.REAL_WORLD_SEND)


def test_permission_policy_blocks_wrong_specialist_identity() -> None:
    permission = get_specialist_permission(SpecialistKind.POLICY)

    with pytest.raises(SpecialistPermissionDenied, match="cannot use policy"):
        permission.require_specialist(SpecialistKind.PLANNING)
