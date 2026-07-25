import pytest

from app.schemas import (
    ApprovalStatus,
    Observation,
    ReActState,
    SpecialistInput,
    SpecialistKind,
    SpecialistPermissionDenied,
    SpecialistPermissionPolicy,
    StopReason,
    WorkflowStatus,
)
from app.workflows import PlanningSpecialistWorkflow, build_planning_workflow


class StubReActWorkflow:
    def __init__(self, result: ReActState) -> None:
        self.result = result
        self.input_state = None

    def invoke(self, state: ReActState):
        self.input_state = state
        return self.result


def planning_input() -> SpecialistInput:
    return SpecialistInput(
        specialist=SpecialistKind.PLANNING,
        request_id="req-planning-contract",
        session_id="session-planning-contract",
        user_message="Plan an outdoor activity.",
        teacher_id="teacher-001",
        class_id="class-001",
        conversation_context="Teacher prefers concise steps.",
    )


def test_planning_workflow_converts_specialist_input_to_react_state() -> None:
    react_workflow = StubReActWorkflow(
        ReActState(
            user_message="Plan an outdoor activity.",
            current_step=2,
            final_answer="Outdoor activity draft.",
            stop_reason=StopReason.COMPLETED,
        )
    )
    workflow = PlanningSpecialistWorkflow(react_workflow, max_steps=7)

    result = workflow.invoke(planning_input())

    assert react_workflow.input_state.user_message == "Plan an outdoor activity."
    assert react_workflow.input_state.teacher_id == "teacher-001"
    assert react_workflow.input_state.class_id == "class-001"
    assert react_workflow.input_state.conversation_context == (
        "Teacher prefers concise steps."
    )
    assert react_workflow.input_state.max_steps == 7
    assert result.specialist is SpecialistKind.PLANNING
    assert result.status is WorkflowStatus.COMPLETED
    assert result.draft is not None
    assert result.draft.content == "Outdoor activity draft."


def test_planning_workflow_maps_react_approval_to_specialist_result() -> None:
    workflow = PlanningSpecialistWorkflow(
        StubReActWorkflow(
            ReActState(
                user_message="Plan and save an activity.",
                current_step=1,
                stop_reason=StopReason.APPROVAL_REQUIRED,
                observations=[
                    Observation(
                        tool_name="save_draft",
                        success=False,
                        error={"code": "permission_denied"},
                    )
                ],
            )
        )
    )

    result = workflow.invoke(planning_input())

    assert result.status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert result.approval.status is ApprovalStatus.REQUIRED
    assert result.trace[0].metadata["observations"][0]["tool_name"] == "save_draft"


def test_planning_workflow_maps_react_failure_to_specialist_error() -> None:
    workflow = PlanningSpecialistWorkflow(
        StubReActWorkflow(
            ReActState(
                user_message="Plan an activity.",
                current_step=7,
                stop_reason=StopReason.MAX_STEPS_REACHED,
            )
        )
    )

    result = workflow.invoke(planning_input())

    assert result.status is WorkflowStatus.FAILED
    assert result.errors[0].code == "max_steps_reached"
    assert result.errors[0].recoverable is True


def test_planning_workflow_rejects_wrong_specialist_kind() -> None:
    workflow = PlanningSpecialistWorkflow(
        StubReActWorkflow(
            ReActState(
                user_message="Draft a learning story.",
                stop_reason=StopReason.COMPLETED,
                final_answer="Draft.",
            )
        )
    )

    with pytest.raises(ValueError, match="specialist=planning"):
        workflow.invoke(
            SpecialistInput(
                specialist=SpecialistKind.DOCUMENTATION,
                request_id="req-wrong-specialist",
                session_id="session-wrong-specialist",
                user_message="Draft a learning story.",
            )
        )


def test_planning_workflow_receives_permission_and_uses_its_step_budget() -> None:
    react_workflow = StubReActWorkflow(
        ReActState(
            user_message="Plan an activity.",
            stop_reason=StopReason.COMPLETED,
            final_answer="Draft.",
        )
    )
    permission = SpecialistPermissionPolicy(
        specialist=SpecialistKind.PLANNING,
        allowed_tool_names=frozenset({"get_class_profile"}),
        max_steps=3,
    )
    workflow = PlanningSpecialistWorkflow(
        react_workflow,
        permission=permission,
    )

    workflow.invoke(planning_input())

    assert workflow.permission == permission
    assert react_workflow.input_state.max_steps == 3


def test_planning_builder_rejects_tool_outside_permission_policy() -> None:
    with pytest.raises(SpecialistPermissionDenied, match="send_family_message"):
        build_planning_workflow(
            allowed_tool_names={"send_family_message"},
        )


def test_planning_builder_rejects_step_budget_expansion() -> None:
    with pytest.raises(SpecialistPermissionDenied, match="7-step budget"):
        build_planning_workflow(max_steps=8)
