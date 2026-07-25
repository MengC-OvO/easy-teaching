"""Family communication specialist skeleton for Week 3 integration."""

from typing import Any, Dict, Mapping, Optional, Union

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.schemas import (
    Draft,
    RiskLevel,
    SafetyFlag,
    SpecialistInput,
    SpecialistKind,
    SpecialistResult,
    TraceEvent,
    WorkflowStatus,
)


class FamilyWorkflowState(BaseModel):
    """Private state used only inside the family communication subgraph."""

    request: SpecialistInput
    result: Optional[SpecialistResult] = None


FamilyWorkflowStateInput = Union[FamilyWorkflowState, Mapping[str, Any]]


def _state_from_input(state: FamilyWorkflowStateInput) -> FamilyWorkflowState:
    if isinstance(state, FamilyWorkflowState):
        return state
    return FamilyWorkflowState.model_validate(state)


def family_draft_node(
    state: FamilyWorkflowStateInput,
) -> Dict[str, SpecialistResult]:
    current_state = _state_from_input(state)
    if current_state.request.specialist is not SpecialistKind.FAMILY:
        raise ValueError("Family workflow requires specialist=family")

    return {
        "result": SpecialistResult(
            specialist=SpecialistKind.FAMILY,
            status=WorkflowStatus.COMPLETED,
            draft=Draft(
                title="Family communication draft",
                content=(
                    "Draft family communication skeleton.\n\n"
                    "Purpose: Pending reviewed context.\n"
                    "Message: Pending de-identified draft generation.\n"
                    "Review: An educator must review this draft before real-world use."
                ),
                is_draft=True,
            ),
            safety_flags=[
                SafetyFlag(
                    code="draft_only",
                    message=(
                        "Family communication is a draft only and must be reviewed "
                        "before real-world use."
                    ),
                    risk_level=RiskLevel.L1_DRAFT,
                )
            ],
            trace=[
                TraceEvent(
                    step="family_draft_skeleton",
                    message="Family communication specialist skeleton completed.",
                    metadata={
                        "request_id": current_state.request.request_id,
                        "implementation": "skeleton",
                    },
                )
            ],
        )
    }


class FamilySpecialistWorkflow:
    """Expose the family communication subgraph through the specialist contract."""

    def __init__(self, graph) -> None:
        self.graph = graph

    def invoke(self, state: SpecialistInput) -> SpecialistResult:
        if state.specialist is not SpecialistKind.FAMILY:
            raise ValueError("Family workflow requires specialist=family")
        output = FamilyWorkflowState.model_validate(
            self.graph.invoke(FamilyWorkflowState(request=state))
        )
        if output.result is None:
            raise ValueError("Family workflow finished without a result")
        return output.result


def build_family_workflow() -> FamilySpecialistWorkflow:
    graph = StateGraph(FamilyWorkflowState)
    graph.add_node("family_draft", family_draft_node)
    graph.set_entry_point("family_draft")
    graph.add_edge("family_draft", END)
    return FamilySpecialistWorkflow(graph.compile())
