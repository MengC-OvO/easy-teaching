"""Planning specialist adapter around the internal ReAct graph."""

import json
from typing import Iterable, Mapping, Optional, Protocol, Union

from pydantic import ValidationError

from app.schemas import (
    ActivityPlan,
    Approval,
    ApprovalStatus,
    Citation,
    get_specialist_permission,
    Draft,
    GraphError,
    ReActState,
    RiskLevel,
    SpecialistInput,
    SpecialistKind,
    SpecialistPermissionDenied,
    SpecialistPermissionPolicy,
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
        permission: Optional[SpecialistPermissionPolicy] = None,
        max_steps: Optional[int] = None,
        required_skill_name: Optional[str] = None,
        validate_activity_plan_output: bool = False,
    ) -> None:
        resolved_permission = _resolve_planning_permission(
            permission,
            max_steps=max_steps,
        )
        self.react_workflow = react_workflow
        self.permission = resolved_permission
        self.max_steps = resolved_permission.max_steps
        self.required_skill_name = required_skill_name
        self.validate_activity_plan_output = validate_activity_plan_output

    def invoke(self, state: SpecialistInput) -> SpecialistResult:
        if state.specialist is not SpecialistKind.PLANNING:
            raise ValueError("Planning workflow requires specialist=planning")

        react_result = self.react_workflow.invoke(
            ReActState(
                user_message=state.user_message,
                teacher_id=state.teacher_id,
                class_id=state.class_id,
                conversation_context=state.conversation_context,
                required_skill_name=self.required_skill_name,
                final_output_schema=(
                    ActivityPlan.model_json_schema()
                    if self.validate_activity_plan_output
                    else {}
                ),
                max_steps=self.max_steps,
            )
        )
        react_state = ReActState.model_validate(react_result)
        trace = [self._trace_event(react_state)]

        if react_state.stop_reason is StopReason.COMPLETED:
            if self.validate_activity_plan_output:
                try:
                    activity_plan = self._parse_activity_plan(
                        react_state.final_answer or ""
                    )
                except (ValidationError, ValueError) as error:
                    trace[0].metadata["output_validation_error"] = str(error)
                    return SpecialistResult(
                        specialist=SpecialistKind.PLANNING,
                        status=WorkflowStatus.FAILED,
                        errors=[
                            GraphError(
                                code="invalid_activity_plan",
                                message=(
                                    "Planning finished with output that does not "
                                    "match the ActivityPlan contract."
                                ),
                                recoverable=True,
                            )
                        ],
                        trace=trace,
                    )
                return SpecialistResult(
                    specialist=SpecialistKind.PLANNING,
                    status=WorkflowStatus.COMPLETED,
                    draft=Draft(
                        title=activity_plan.title,
                        content=activity_plan.model_dump_json(indent=2),
                        is_draft=True,
                    ),
                    citations=self._activity_plan_citations(activity_plan),
                    trace=trace,
                )
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
                        StopReason.SKILL_REQUIRED,
                        StopReason.SKILL_REQUIREMENTS_MISSING,
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
                "loaded_skill_name": (
                    state.loaded_skill.manifest.name
                    if state.loaded_skill is not None
                    else None
                ),
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

    def _parse_activity_plan(self, final_answer: str) -> ActivityPlan:
        candidate = final_answer.strip()
        if candidate.startswith("```") and candidate.endswith("```"):
            lines = candidate.splitlines()
            if len(lines) >= 3:
                candidate = "\n".join(lines[1:-1]).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as error:
            raise ValueError("final answer is not valid JSON") from error
        return ActivityPlan.model_validate(payload)

    def _activity_plan_citations(
        self,
        plan: ActivityPlan,
    ) -> list[Citation]:
        citations = []
        seen = set()
        for alignment in plan.eylf_alignments:
            for citation in alignment.citations:
                key = citation.model_dump_json()
                if key not in seen:
                    citations.append(citation)
                    seen.add(key)
        return citations


def build_planning_workflow(
    *,
    agent: Optional[ReActAgentProtocol] = None,
    registry: Optional[ToolRegistry] = None,
    allowed_tool_names: Optional[Iterable[str]] = None,
    approved: bool = False,
    max_steps: Optional[int] = None,
    permission: Optional[SpecialistPermissionPolicy] = None,
    required_skill_name: Optional[str] = "activity_planning",
) -> PlanningSpecialistWorkflow:
    resolved_permission = _resolve_planning_permission(
        permission,
        allowed_tool_names=allowed_tool_names,
        max_steps=max_steps,
    )
    if required_skill_name is not None:
        resolved_permission.require_tool("load_skill")
    react_workflow = build_react_graph(
        agent=agent,
        registry=registry,
        allowed_tool_names=resolved_permission.allowed_tool_names,
        approved=approved,
        required_skill_name=required_skill_name,
    )
    return PlanningSpecialistWorkflow(
        react_workflow,
        permission=resolved_permission,
        required_skill_name=required_skill_name,
        validate_activity_plan_output=required_skill_name is not None,
    )


def _resolve_planning_permission(
    permission: Optional[SpecialistPermissionPolicy],
    *,
    allowed_tool_names: Optional[Iterable[str]] = None,
    max_steps: Optional[int] = None,
) -> SpecialistPermissionPolicy:
    resolved = permission or get_specialist_permission(SpecialistKind.PLANNING)
    resolved.require_specialist(SpecialistKind.PLANNING)

    data = resolved.model_dump()
    if allowed_tool_names is not None:
        requested_tool_names = frozenset(allowed_tool_names)
        unauthorized_tools = requested_tool_names - resolved.allowed_tool_names
        if unauthorized_tools:
            tool_names = ", ".join(sorted(unauthorized_tools))
            raise SpecialistPermissionDenied(
                f"planning specialist cannot use tools: {tool_names}"
            )
        data["allowed_tool_names"] = requested_tool_names
    if max_steps is not None:
        if max_steps > resolved.max_steps:
            raise SpecialistPermissionDenied(
                f"planning specialist cannot exceed its "
                f"{resolved.max_steps}-step budget"
            )
        data["max_steps"] = max_steps
    return SpecialistPermissionPolicy.model_validate(data)
