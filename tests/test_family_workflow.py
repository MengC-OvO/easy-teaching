import pytest

from app.schemas import (
    RiskLevel,
    SpecialistInput,
    SpecialistKind,
    WorkflowStatus,
)
from app.workflows import build_family_workflow


def family_input() -> SpecialistInput:
    return SpecialistInput(
        specialist=SpecialistKind.FAMILY,
        request_id="req-family",
        session_id="session-family",
        user_message="Draft a family update from a synthetic classroom event.",
        teacher_id="teacher-001",
        class_id="class-001",
        conversation_context="Teacher prefers concise family updates.",
    )


def test_family_subgraph_returns_a_safe_draft_skeleton() -> None:
    workflow = build_family_workflow()

    result = workflow.invoke(family_input())

    assert result.specialist is SpecialistKind.FAMILY
    assert result.status is WorkflowStatus.COMPLETED
    assert result.draft is not None
    assert result.draft.title == "Family communication draft"
    assert result.draft.is_draft is True
    assert "Pending de-identified draft generation" in result.draft.content
    assert result.safety_flags[0].code == "draft_only"
    assert result.safety_flags[0].risk_level is RiskLevel.L1_DRAFT
    assert result.trace[0].step == "family_draft_skeleton"


def test_family_skeleton_does_not_copy_unprocessed_private_text() -> None:
    workflow = build_family_workflow()
    specialist_input = family_input().model_copy(
        update={"user_message": "Parent Jordan Example asked about child Alex Example."}
    )

    result = workflow.invoke(specialist_input)

    assert result.draft is not None
    assert "Jordan Example" not in result.draft.content
    assert "Alex Example" not in result.draft.content


def test_family_subgraph_rejects_wrong_specialist_kind() -> None:
    workflow = build_family_workflow()

    with pytest.raises(ValueError, match="specialist=family"):
        workflow.invoke(
            SpecialistInput(
                specialist=SpecialistKind.POLICY,
                request_id="req-wrong-family",
                session_id="session-wrong-family",
                user_message="What does the policy say?",
            )
        )
