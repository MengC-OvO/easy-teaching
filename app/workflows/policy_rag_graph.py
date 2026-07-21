from typing import Any, Dict, Mapping, Optional, Protocol, Union

from langgraph.graph import END, StateGraph

from app.schemas import (
    Citation,
    Draft,
    GraphError,
    GraphState,
    PolicyRAGResult,
    PolicyRAGStatus,
    TraceEvent,
    WorkflowStatus,
)
from app.services import ChatCompletionsModelProvider, ModelProviderError, PolicyRAGService


GraphStateInput = Union[GraphState, Mapping[str, Any]]


class PolicyRAGServiceProtocol(Protocol):
    def answer(self, question: str) -> PolicyRAGResult:
        ...


def _state_from_input(state: GraphStateInput) -> GraphState:
    if isinstance(state, GraphState):
        return state
    return GraphState.model_validate(state)


def build_policy_rag_node(service: PolicyRAGServiceProtocol):
    def policy_rag_node(state: GraphStateInput) -> Dict[str, Any]:
        current_state = _state_from_input(state)
        try:
            result = service.answer(current_state.user_message)
        except ModelProviderError as error:
            return {
                "workflow_status": WorkflowStatus.FAILED,
                "errors": [
                    GraphError(
                        code=error.code.value,
                        message=error.message,
                        recoverable=error.recoverable,
                    )
                ],
                "trace": [
                    TraceEvent(
                        step="policy_rag",
                        message="Policy RAG workflow failed.",
                        metadata=error.to_dict(),
                    )
                ],
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

        if result.status is PolicyRAGStatus.ANSWERED:
            return {
                "workflow_status": WorkflowStatus.COMPLETED,
                "draft": Draft(
                    title="Policy answer draft",
                    content=result.answer or "",
                    is_draft=True,
                ),
                "citations": [_to_graph_citation(citation) for citation in result.citations],
                "trace": trace,
            }

        if result.status is PolicyRAGStatus.NEEDS_CLARIFICATION:
            return {
                "workflow_status": WorkflowStatus.ROUTED,
                "needs_clarification": True,
                "clarification_question": result.clarification_question,
                "trace": trace,
            }

        return {
            "workflow_status": WorkflowStatus.FAILED,
            "errors": [
                GraphError(
                    code=result.status.value,
                    message=result.refusal_reason or "Policy RAG could not answer.",
                    recoverable=result.status is PolicyRAGStatus.EVIDENCE_CONFLICT,
                )
            ],
            "citations": [_to_graph_citation(citation) for citation in result.citations],
            "trace": trace,
        }

    return policy_rag_node


def _to_graph_citation(citation) -> Citation:
    return Citation(
        source=citation.source_id,
        title=citation.title,
        section=citation.section,
        page=citation.page,
        url=citation.uri,
    )


def build_policy_rag_graph(service: Optional[PolicyRAGServiceProtocol] = None):
    resolved_service = service or PolicyRAGService(
        model_provider=ChatCompletionsModelProvider(),
    )
    graph = StateGraph(GraphState)
    graph.add_node("policy_rag", build_policy_rag_node(resolved_service))
    graph.set_entry_point("policy_rag")
    graph.add_edge("policy_rag", END)
    return graph.compile()
