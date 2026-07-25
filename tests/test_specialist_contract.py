import pytest
from pydantic import ValidationError

from app.schemas import (
    Approval,
    ApprovalStatus,
    Citation,
    Draft,
    GraphError,
    GraphState,
    RiskLevel,
    SpecialistInput,
    SpecialistKind,
    SpecialistResult,
    TraceEvent,
    WorkflowStatus,
)
from app.workflows.specialist import SpecialistWorkflowProtocol


def test_specialist_input_projects_only_bounded_main_graph_state() -> None:
    state = GraphState(
        request_id="req-specialist-001",
        session_id="session-specialist-001",
        thread_id="thread-specialist-001",
        user_message="Plan an outdoor activity.",
        teacher_id="teacher-001",
        class_id="class-001",
        citations=[Citation(source="old-source")],
    )

    specialist_input = SpecialistInput.from_graph_state(
        state,
        specialist=SpecialistKind.PLANNING,
        conversation_context="Teacher prefers concise steps.",
    )

    assert specialist_input.specialist is SpecialistKind.PLANNING
    assert specialist_input.request_id == state.request_id
    assert specialist_input.thread_id == state.thread_id
    assert specialist_input.teacher_id == state.teacher_id
    assert specialist_input.conversation_context == "Teacher prefers concise steps."
    assert "citations" not in SpecialistInput.model_fields


def test_specialist_result_maps_shared_output_to_graph_update() -> None:
    result = SpecialistResult(
        specialist=SpecialistKind.POLICY,
        status=WorkflowStatus.COMPLETED,
        draft=Draft(title="Policy answer", content="Evidence-based draft."),
        citations=[Citation(source="eylf-v2", page=8)],
        trace=[TraceEvent(step="policy", message="Policy workflow completed.")],
    )

    update = result.to_graph_update()

    assert update["workflow_status"] is WorkflowStatus.COMPLETED
    assert update["draft"].content == "Evidence-based draft."
    assert update["citations"][0].source == "eylf-v2"
    assert update["trace"][0].step == "policy"


def test_specialist_result_requires_question_when_clarification_is_needed() -> None:
    with pytest.raises(ValidationError, match="must include a question"):
        SpecialistResult(
            specialist=SpecialistKind.DOCUMENTATION,
            status=WorkflowStatus.ROUTED,
            needs_clarification=True,
        )


def test_specialist_result_requires_approval_payload_when_waiting() -> None:
    with pytest.raises(ValidationError, match="must require approval"):
        SpecialistResult(
            specialist=SpecialistKind.FAMILY,
            status=WorkflowStatus.WAITING_FOR_APPROVAL,
        )

    result = SpecialistResult(
        specialist=SpecialistKind.FAMILY,
        status=WorkflowStatus.WAITING_FOR_APPROVAL,
        approval=Approval(
            status=ApprovalStatus.REQUIRED,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
            reason="Teacher review is required.",
        ),
    )

    assert result.approval.status is ApprovalStatus.REQUIRED


def test_specialist_result_requires_error_when_failed() -> None:
    with pytest.raises(ValidationError, match="must include an error"):
        SpecialistResult(
            specialist=SpecialistKind.PLANNING,
            status=WorkflowStatus.FAILED,
        )

    result = SpecialistResult(
        specialist=SpecialistKind.PLANNING,
        status=WorkflowStatus.FAILED,
        errors=[
            GraphError(
                code="tool_error",
                message="Planning tool failed.",
                recoverable=True,
            )
        ],
    )

    assert result.errors[0].code == "tool_error"


def test_specialist_workflow_protocol_describes_invoke_contract() -> None:
    class StubSpecialistWorkflow:
        def invoke(self, state: SpecialistInput) -> SpecialistResult:
            return SpecialistResult(
                specialist=state.specialist,
                status=WorkflowStatus.COMPLETED,
                draft=Draft(content="Draft output."),
            )

    workflow: SpecialistWorkflowProtocol = StubSpecialistWorkflow()
    result = workflow.invoke(
        SpecialistInput(
            specialist=SpecialistKind.DOCUMENTATION,
            request_id="req-specialist-002",
            session_id="session-specialist-002",
            user_message="Draft a learning story.",
        )
    )

    assert isinstance(result, SpecialistResult)
    assert result.specialist is SpecialistKind.DOCUMENTATION
