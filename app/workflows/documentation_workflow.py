"""Documentation specialist subgraph behind the shared specialist contract."""

from typing import Any, Dict, Mapping, Optional, Protocol, Union

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.schemas import (
    Approval,
    ApprovalStatus,
    DeidentifiedObservation,
    Draft,
    get_specialist_permission,
    GraphError,
    LearningRecordDraft,
    RiskLevel,
    SpecialistInput,
    SpecialistKind,
    SpecialistPermissionPolicy,
    SpecialistResult,
    TraceEvent,
    WorkflowStatus,
)
from app.services import (
    ChatCompletionsModelProvider,
    LearningRecordDraftingService,
    ModelProviderError,
)


class DocumentationWorkflowState(BaseModel):
    """Private state used only while the documentation subgraph is running."""

    request: SpecialistInput
    result: Optional[SpecialistResult] = None


DocumentationWorkflowStateInput = Union[
    DocumentationWorkflowState,
    Mapping[str, Any],
]


class DocumentationDraftingProtocol(Protocol):
    def create_draft(
        self,
        observation_text: str,
    ) -> tuple[LearningRecordDraft, DeidentifiedObservation]:
        ...


def _state_from_input(
    state: DocumentationWorkflowStateInput,
) -> DocumentationWorkflowState:
    if isinstance(state, DocumentationWorkflowState):
        return state
    return DocumentationWorkflowState.model_validate(state)


def build_documentation_draft_node(service: DocumentationDraftingProtocol):
    """Draft from redacted text and stop before any controlled write."""

    def documentation_draft_node(
        state: DocumentationWorkflowStateInput,
    ) -> Dict[str, SpecialistResult]:
        current_state = _state_from_input(state)
        request = current_state.request
        if request.specialist is not SpecialistKind.DOCUMENTATION:
            raise ValueError("Documentation workflow requires specialist=documentation")

        try:
            draft, deidentified = service.create_draft(request.user_message)
        except (ModelProviderError, ValueError) as error:
            metadata = (
                error.safe_metadata()
                if isinstance(error, ModelProviderError)
                else {}
            )
            return {
                "result": SpecialistResult(
                    specialist=SpecialistKind.DOCUMENTATION,
                    status=WorkflowStatus.FAILED,
                    errors=[
                        GraphError(
                            code=(
                                error.code.value
                                if isinstance(error, ModelProviderError)
                                else "documentation_draft_invalid"
                            ),
                            message=(
                                error.message
                                if isinstance(error, ModelProviderError)
                                else str(error)
                            ),
                            recoverable=(
                                error.recoverable
                                if isinstance(error, ModelProviderError)
                                else True
                            ),
                        )
                    ],
                    trace=[
                        TraceEvent(
                            step="documentation_draft",
                            message="Documentation specialist draft generation failed.",
                            metadata=metadata,
                        )
                    ],
                )
            }

        return {
            "result": SpecialistResult(
                specialist=SpecialistKind.DOCUMENTATION,
                status=WorkflowStatus.WAITING_FOR_APPROVAL,
                draft=Draft(
                    title="Learning record draft",
                    content=draft.model_dump_json(indent=2),
                    is_draft=True,
                ),
                approval=Approval(
                    status=ApprovalStatus.REQUIRED,
                    risk_level=RiskLevel.L2_CONTROLLED_WRITE,
                    reason=(
                        "Teacher review is required before this learning record "
                        "can be saved."
                    ),
                ),
                trace=[
                    TraceEvent(
                        step="documentation_draft",
                        message=(
                            "Generated a de-identified learning-record draft "
                            "and is waiting for teacher review."
                        ),
                        metadata={
                            "request_id": request.request_id,
                            "redacted_types": [
                                pii_type.value
                                for pii_type in deidentified.redacted_types
                            ],
                            "replacement_count": deidentified.replacement_count,
                        },
                    )
                ],
            )
        }

    return documentation_draft_node


class DocumentationSpecialistWorkflow:
    """Expose the internal documentation graph through the specialist contract."""

    def __init__(
        self,
        graph,
        *,
        permission: Optional[SpecialistPermissionPolicy] = None,
    ) -> None:
        resolved_permission = permission or get_specialist_permission(
            SpecialistKind.DOCUMENTATION
        )
        if resolved_permission.specialist is not SpecialistKind.DOCUMENTATION:
            raise ValueError(
                "Documentation workflow requires a documentation permission policy"
            )
        self.graph = graph
        self.permission = resolved_permission

    def invoke(self, state: SpecialistInput) -> SpecialistResult:
        if state.specialist is not SpecialistKind.DOCUMENTATION:
            raise ValueError("Documentation workflow requires specialist=documentation")
        output = DocumentationWorkflowState.model_validate(
            self.graph.invoke(DocumentationWorkflowState(request=state))
        )
        if output.result is None:
            raise ValueError("Documentation workflow finished without a result")
        return output.result


def build_documentation_workflow(
    service: Optional[DocumentationDraftingProtocol] = None,
    *,
    permission: Optional[SpecialistPermissionPolicy] = None,
) -> DocumentationSpecialistWorkflow:
    """Build the documentation specialist subgraph used by the main graph."""
    resolved_service = service or LearningRecordDraftingService(
        model_provider=ChatCompletionsModelProvider(),
    )
    graph = StateGraph(DocumentationWorkflowState)
    graph.add_node("documentation_draft", build_documentation_draft_node(resolved_service))
    graph.set_entry_point("documentation_draft")
    graph.add_edge("documentation_draft", END)
    return DocumentationSpecialistWorkflow(graph.compile(), permission=permission)
