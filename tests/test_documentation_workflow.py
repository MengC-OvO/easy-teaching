import pytest

from app.schemas import (
    SpecialistInput,
    SpecialistKind,
    WorkflowStatus,
)
from app.workflows import build_documentation_workflow


def documentation_input() -> SpecialistInput:
    return SpecialistInput(
        specialist=SpecialistKind.DOCUMENTATION,
        request_id="req-documentation",
        session_id="session-documentation",
        user_message="Write a learning story from a synthetic observation.",
        teacher_id="teacher-001",
        class_id="class-001",
        conversation_context="Teacher prefers concise learning records.",
    )


def test_documentation_subgraph_returns_a_safe_draft_skeleton() -> None:
    workflow = build_documentation_workflow()

    result = workflow.invoke(documentation_input())

    assert result.specialist is SpecialistKind.DOCUMENTATION
    assert result.status is WorkflowStatus.COMPLETED
    assert result.draft is not None
    assert result.draft.title == "Learning record draft"
    assert result.draft.is_draft is True
    assert "Pending de-identified observation processing" in result.draft.content
    assert result.trace[0].step == "documentation_skeleton"
    assert result.trace[0].metadata["implementation"] == "skeleton"


def test_documentation_skeleton_does_not_copy_unprocessed_observation_text() -> None:
    workflow = build_documentation_workflow()
    specialist_input = documentation_input().model_copy(
        update={"user_message": "Synthetic child name: Alex Example."}
    )

    result = workflow.invoke(specialist_input)

    assert result.draft is not None
    assert "Alex Example" not in result.draft.content


def test_documentation_subgraph_rejects_wrong_specialist_kind() -> None:
    workflow = build_documentation_workflow()

    with pytest.raises(ValueError, match="specialist=documentation"):
        workflow.invoke(
            SpecialistInput(
                specialist=SpecialistKind.FAMILY,
                request_id="req-wrong-documentation",
                session_id="session-wrong-documentation",
                user_message="Draft a family update.",
            )
        )
