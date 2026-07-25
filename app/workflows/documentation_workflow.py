"""Documentation specialist skeleton for Week 3 workflow integration."""

from typing import Any, Dict, Mapping, Optional, Union

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.schemas import (
    Draft,
    SpecialistInput,
    SpecialistKind,
    SpecialistResult,
    TraceEvent,
    WorkflowStatus,
)


class DocumentationWorkflowState(BaseModel):
    """Private state used only inside the documentation subgraph."""

    request: SpecialistInput
    result: Optional[SpecialistResult] = None


DocumentationWorkflowStateInput = Union[
    DocumentationWorkflowState,
    Mapping[str, Any],
]


def _state_from_input(
    state: DocumentationWorkflowStateInput,
) -> DocumentationWorkflowState:
    if isinstance(state, DocumentationWorkflowState):
        return state
    return DocumentationWorkflowState.model_validate(state)


def documentation_skeleton_node(
    state: DocumentationWorkflowStateInput,
) -> Dict[str, SpecialistResult]:
    current_state = _state_from_input(state)
    if current_state.request.specialist is not SpecialistKind.DOCUMENTATION:
        raise ValueError("Documentation workflow requires specialist=documentation")

    return {
        "result": SpecialistResult(
            specialist=SpecialistKind.DOCUMENTATION,
            status=WorkflowStatus.COMPLETED,
            draft=Draft(
                title="Learning record draft",
                content=(
                    "Draft documentation skeleton.\n\n"
                    "Observation: Pending de-identified observation processing.\n"
                    "Reflection: Pending documentation specialist generation.\n"
                    "Possible next steps: Pending educator review."
                ),
                is_draft=True,
            ),
            trace=[
                TraceEvent(
                    step="documentation_skeleton",
                    message="Documentation specialist skeleton completed.",
                    metadata={
                        "request_id": current_state.request.request_id,
                        "implementation": "skeleton",
                    },
                )
            ],
        )
    }


class DocumentationSpecialistWorkflow:
    """Expose the documentation subgraph through the specialist contract."""

    def __init__(self, graph) -> None:
        self.graph = graph

    def invoke(self, state: SpecialistInput) -> SpecialistResult:
        if state.specialist is not SpecialistKind.DOCUMENTATION:
            raise ValueError("Documentation workflow requires specialist=documentation")
        output = DocumentationWorkflowState.model_validate(
            self.graph.invoke(DocumentationWorkflowState(request=state))
        )
        if output.result is None:
            raise ValueError("Documentation workflow finished without a result")
        return output.result


def build_documentation_workflow() -> DocumentationSpecialistWorkflow:
    graph = StateGraph(DocumentationWorkflowState)
    graph.add_node("documentation", documentation_skeleton_node)
    graph.set_entry_point("documentation")
    graph.add_edge("documentation", END)
    return DocumentationSpecialistWorkflow(graph.compile())
