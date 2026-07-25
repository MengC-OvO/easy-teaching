from typing import Any, Dict, Mapping, Optional, Protocol, Union

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from app.schemas import (
    Citation,
    Draft,
    GraphError,
    PolicyRAGResult,
    PolicyRAGStatus,
    SpecialistInput,
    SpecialistKind,
    SpecialistResult,
    TraceEvent,
    WorkflowStatus,
)
from app.services import (
    ChatCompletionsModelProvider,
    ModelProviderError,
    PolicyRAGService,
)


class PolicyWorkflowState(BaseModel):
    """Private state used only while the policy specialist graph is running."""

    request: SpecialistInput
    result: Optional[SpecialistResult] = None


PolicyWorkflowStateInput = Union[PolicyWorkflowState, Mapping[str, Any]]


class PolicyRAGServiceProtocol(Protocol):
    def answer(
        self,
        question: str,
        *,
        conversation_context: str = "",
    ) -> PolicyRAGResult:
        ...


def _state_from_input(state: PolicyWorkflowStateInput) -> PolicyWorkflowState:
    if isinstance(state, PolicyWorkflowState):
        return state
    return PolicyWorkflowState.model_validate(state)


def build_policy_rag_node(service: PolicyRAGServiceProtocol):
    def policy_rag_node(state: PolicyWorkflowStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        request = current_state.request
        try:
            result = service.answer(
                request.user_message,
                conversation_context=request.conversation_context,
            )
        except ModelProviderError as error:
            return {
                "result": SpecialistResult(
                    specialist=SpecialistKind.POLICY,
                    status=WorkflowStatus.FAILED,
                    errors=[
                        GraphError(
                            code=error.code.value,
                            message=error.message,
                            recoverable=error.recoverable,
                        )
                    ],
                    trace=[
                        TraceEvent(
                            step="policy_rag",
                            message="Policy RAG workflow failed.",
                            metadata=error.to_dict(),
                        )
                    ],
                )
            }

        trace = [
            TraceEvent(
                step="policy_rag",
                message="Policy RAG workflow completed.",
                metadata={
                    "status": result.status.value,
                    "evidence_count": len(result.evidence),
                    "citation_count": len(result.citations),
                    "retrieval": result.retrieval.stats.model_dump(mode="json"),
                },
            )
        ]
        citations = [_to_graph_citation(citation) for citation in result.citations]

        if result.status is PolicyRAGStatus.ANSWERED:
            specialist_result = SpecialistResult(
                specialist=SpecialistKind.POLICY,
                status=WorkflowStatus.COMPLETED,
                draft=Draft(
                    title="Policy answer draft",
                    content=result.answer or "",
                    is_draft=True,
                ),
                citations=citations,
                trace=trace,
            )
        elif result.status is PolicyRAGStatus.NEEDS_CLARIFICATION:
            specialist_result = SpecialistResult(
                specialist=SpecialistKind.POLICY,
                status=WorkflowStatus.ROUTED,
                needs_clarification=True,
                clarification_question=result.clarification_question,
                trace=trace,
            )
        else:
            specialist_result = SpecialistResult(
                specialist=SpecialistKind.POLICY,
                status=WorkflowStatus.FAILED,
                errors=[
                    GraphError(
                        code=result.status.value,
                        message=result.refusal_reason or "Policy RAG could not answer.",
                        recoverable=result.status is PolicyRAGStatus.EVIDENCE_CONFLICT,
                    )
                ],
                citations=citations,
                trace=trace,
            )
        return {"result": specialist_result}

    return policy_rag_node


def _to_graph_citation(citation) -> Citation:
    return Citation(
        source=citation.source_id,
        title=citation.title,
        section=citation.section,
        page=citation.page,
        url=citation.uri,
    )


class PolicySpecialistWorkflow:
    """Expose the internal policy RAG graph through the specialist contract."""

    def __init__(self, graph) -> None:
        self.graph = graph

    def invoke(self, state: SpecialistInput) -> SpecialistResult:
        if state.specialist is not SpecialistKind.POLICY:
            raise ValueError("Policy workflow requires specialist=policy")
        output = PolicyWorkflowState.model_validate(
            self.graph.invoke(PolicyWorkflowState(request=state))
        )
        if output.result is None:
            raise ValueError("Policy workflow finished without a result")
        return output.result


def build_policy_rag_graph(
    service: Optional[PolicyRAGServiceProtocol] = None,
) -> PolicySpecialistWorkflow:
    resolved_service = service or PolicyRAGService(
        model_provider=ChatCompletionsModelProvider(),
    )
    graph = StateGraph(PolicyWorkflowState)
    graph.add_node("policy_rag", build_policy_rag_node(resolved_service))
    graph.set_entry_point("policy_rag")
    graph.add_edge("policy_rag", END)
    return PolicySpecialistWorkflow(graph.compile())
