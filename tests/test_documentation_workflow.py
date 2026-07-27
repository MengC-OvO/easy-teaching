import json

import pytest

from app.schemas import (
    DeidentifiedObservation,
    LearningRecordDraft,
    PIIType,
    SpecialistInput,
    SpecialistKind,
    SpecialistPermissionPolicy,
    WorkflowStatus,
)
from app.workflows import build_documentation_workflow


class StubDocumentationDraftingService:
    def __init__(self) -> None:
        self.observation_texts = []

    def create_draft(self, observation_text: str):
        self.observation_texts.append(observation_text)
        return (
            LearningRecordDraft(
                observation_summary="[PERSON_NAME_1] persisted with block balancing.",
                learning_analysis="The observation suggests sustained engagement.",
                possible_next_steps=["Offer varied balancing materials."],
            ),
            DeidentifiedObservation(
                safe_text="Child named [PERSON_NAME_1] persisted with blocks.",
                redacted_types=[PIIType.PERSON_NAME],
                replacement_count=1,
            ),
        )


def documentation_input() -> SpecialistInput:
    return SpecialistInput(
        specialist=SpecialistKind.DOCUMENTATION,
        request_id="req-documentation",
        session_id="session-documentation",
        user_message="Child named Alex Example persisted with blocks.",
        teacher_id="teacher-001",
        class_id="class-001",
        conversation_context="Teacher prefers concise learning records.",
    )


def test_documentation_subgraph_returns_a_draft_waiting_for_teacher_approval() -> None:
    service = StubDocumentationDraftingService()
    workflow = build_documentation_workflow(service)

    result = workflow.invoke(documentation_input())

    assert service.observation_texts == [
        "Child named Alex Example persisted with blocks."
    ]
    assert result.specialist is SpecialistKind.DOCUMENTATION
    assert result.status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert result.approval.status.value == "required"
    assert result.draft is not None
    assert result.draft.title == "Learning record draft"
    assert result.draft.is_draft is True
    assert json.loads(result.draft.content) == {
        "observation_summary": "[PERSON_NAME_1] persisted with block balancing.",
        "learning_analysis": "The observation suggests sustained engagement.",
        "possible_next_steps": ["Offer varied balancing materials."],
        "is_draft": True,
    }
    assert result.trace[0].step == "documentation_draft"
    assert result.trace[0].metadata["redacted_types"] == ["person_name"]
    assert "Alex Example" not in result.draft.content


def test_documentation_subgraph_returns_failure_when_drafting_service_fails() -> None:
    class BrokenDocumentationDraftingService:
        def create_draft(self, observation_text: str):
            raise ValueError("Model provider did not return a learning-record draft")

    workflow = build_documentation_workflow(BrokenDocumentationDraftingService())

    result = workflow.invoke(documentation_input())

    assert result.status is WorkflowStatus.FAILED
    assert result.errors[0].code == "documentation_draft_invalid"
    assert result.trace[0].step == "documentation_draft"


def test_documentation_subgraph_rejects_wrong_specialist_kind() -> None:
    workflow = build_documentation_workflow(StubDocumentationDraftingService())

    with pytest.raises(ValueError, match="specialist=documentation"):
        workflow.invoke(
            SpecialistInput(
                specialist=SpecialistKind.FAMILY,
                request_id="req-wrong-documentation",
                session_id="session-wrong-documentation",
                user_message="Draft a family update.",
            )
        )


def test_documentation_workflow_receives_its_permission_policy() -> None:
    permission = SpecialistPermissionPolicy(
        specialist=SpecialistKind.DOCUMENTATION,
        max_steps=1,
    )

    workflow = build_documentation_workflow(
        StubDocumentationDraftingService(),
        permission=permission,
    )

    assert workflow.permission == permission
