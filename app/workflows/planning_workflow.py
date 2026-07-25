"""Planning specialist adapter around the internal ReAct graph."""

from typing import Iterable, Mapping, Optional, Protocol, Union

from app.schemas import (
    Approval,
    ApprovalStatus,
    Draft,
    GraphError,
    ReActState,
    RiskLevel,
    SpecialistInput,
    SpecialistKind,
    SpecialistResult,
    StopReason,
    TraceEvent,
    WorkflowStatus,
)
from app.tools import ToolRegistry
from app.workflows.react_graph import ReActAgentProtocol, build_react_graph


ReActWorkflowOutput = Union[ReActState, Mapping[str, object]]


class ReActWorkflowProtocol(Protocol):
    def invoke(self, state: ReActState) -> ReActWorkflowOutput:
        ...


class PlanningSpecialistWorkflow:
    """Expose a ReAct planning graph through the shared specialist contract."""

    def __init__(
        self,
        react_workflow: ReActWorkflowProtocol,
        *,
        max_steps: int = 7,
    ) -> None:
        self.react_workflow = react_workflow
        self.max_steps = max_steps

    def invoke(self, state: SpecialistInput) -> SpecialistResult:
        if state.specialist is not SpecialistKind.PLANNING:
            raise ValueError("Planning workflow requires specialist=planning")

        react_result = self.react_workflow.invoke(
            ReActState(
                user_message=state.user_message,
                teacher_id=state.teacher_id,
                class_id=state.class_id,
                conversation_context=state.conversation_context,
                max_steps=self.max_steps,
            )
        )
        react_state = ReActState.model_validate(react_result)
        trace = [self._trace_event(react_state)]

        if react_state.stop_reason is StopReason.COMPLETED:
            return SpecialistResult(
                specialist=SpecialistKind.PLANNING,
                status=WorkflowStatus.COMPLETED,
                draft=Draft(
                    title="Activity planning draft",
                    content=react_state.final_answer or "",
                    is_draft=True,
                ),
                trace=trace,
            )

        if react_state.stop_reason is StopReason.APPROVAL_REQUIRED:
            return SpecialistResult(
                specialist=SpecialistKind.PLANNING,
                status=WorkflowStatus.WAITING_FOR_APPROVAL,
                approval=Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason="A controlled write tool requires teacher approval.",
                ),
                trace=trace,
            )

        return SpecialistResult(
            specialist=SpecialistKind.PLANNING,
            status=WorkflowStatus.FAILED,
            errors=[
                GraphError(
                    code=react_state.stop_reason.value,
                    message="Activity planning ReAct workflow stopped before completion.",
                    recoverable=react_state.stop_reason
                    in {
                        StopReason.MAX_STEPS_REACHED,
                        StopReason.TOOL_ERROR,
                        StopReason.MODEL_ERROR,
                    },
                )
            ],
            trace=trace,
        )

    def _trace_event(self, state: ReActState) -> TraceEvent:
        return TraceEvent(
            step="planning_react",
            message="Activity planning ReAct workflow completed.",
            metadata={
                "stop_reason": state.stop_reason.value,
                "current_step": state.current_step,
                "observations": [
                    {
                        "tool_name": observation.tool_name,
                        "success": observation.success,
                        "error_code": (
                            observation.error.get("code")
                            if observation.error
                            else None
                        ),
                    }
                    for observation in state.observations
                ],
            },
        )


def build_planning_workflow(
    *,
    agent: Optional[ReActAgentProtocol] = None,
    registry: Optional[ToolRegistry] = None,
    allowed_tool_names: Optional[Iterable[str]] = None,
    approved: bool = False,
    max_steps: int = 7,
) -> PlanningSpecialistWorkflow:
    react_workflow = build_react_graph(
        agent=agent,
        registry=registry,
        allowed_tool_names=allowed_tool_names,
        approved=approved,
    )
    return PlanningSpecialistWorkflow(react_workflow, max_steps=max_steps)
