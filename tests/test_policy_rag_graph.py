from app.schemas import (
    CitationMetadata,
    GraphState,
    KnowledgeSourceType,
    PolicyRAGResult,
    PolicyRAGStatus,
    RetrievalResult,
    RetrievalStats,
    WorkflowStatus,
)
from app.workflows.policy_rag_graph import build_policy_rag_graph


class StubPolicyRAGService:
    def __init__(self, result: PolicyRAGResult) -> None:
        self.result = result
        self.question = None

    def answer(
        self,
        question: str,
        *,
        conversation_context: str = "",
    ) -> PolicyRAGResult:
        self.question = question
        return self.result


def empty_retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="What does policy say?",
        chunks=[],
        stats=RetrievalStats(
            requested_top_k=5,
            raw_result_count=0,
            deduplicated_count=0,
            returned_count=0,
        ),
    )


def test_policy_rag_graph_maps_answer_to_draft_and_citations() -> None:
    citation = CitationMetadata(
        source_id="eylf-v2",
        source_type=KnowledgeSourceType.OFFICIAL,
        title="EYLF V2.0",
        version="2.0-2022",
        page=21,
        uri="https://example.test/eylf.pdf",
    )
    service = StubPolicyRAGService(
        PolicyRAGResult(
            status=PolicyRAGStatus.ANSWERED,
            question="What does policy say?",
            answer="Use evidence cautiously [E1].",
            citations=[citation],
            retrieval=empty_retrieval(),
        )
    )
    graph = build_policy_rag_graph(service)

    final_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-policy",
                session_id="session-policy",
                user_message="What does policy say?",
            )
        )
    )

    assert final_state.workflow_status is WorkflowStatus.COMPLETED
    assert final_state.draft is not None
    assert final_state.draft.title == "Policy answer draft"
    assert final_state.draft.content == "Use evidence cautiously [E1]."
    assert final_state.citations[0].source == "eylf-v2"
    assert final_state.citations[0].page == 21
    assert final_state.trace[-1].step == "policy_rag"
    assert service.question == "What does policy say?"


def test_policy_rag_graph_maps_empty_retrieval_to_clarification() -> None:
    service = StubPolicyRAGService(
        PolicyRAGResult(
            status=PolicyRAGStatus.NEEDS_CLARIFICATION,
            question="What policy?",
            clarification_question="Which policy area should I search?",
            retrieval=empty_retrieval(),
        )
    )
    graph = build_policy_rag_graph(service)

    final_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-policy",
                session_id="session-policy",
                user_message="What policy?",
            )
        )
    )

    assert final_state.needs_clarification is True
    assert final_state.clarification_question == "Which policy area should I search?"
    assert final_state.trace[-1].metadata["status"] == "needs_clarification"


def test_policy_rag_graph_maps_conflict_to_error() -> None:
    service = StubPolicyRAGService(
        PolicyRAGResult(
            status=PolicyRAGStatus.EVIDENCE_CONFLICT,
            question="What policy?",
            refusal_reason="Conflicting source versions.",
            retrieval=empty_retrieval(),
        )
    )
    graph = build_policy_rag_graph(service)

    final_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-policy",
                session_id="session-policy",
                user_message="What policy?",
            )
        )
    )

    assert final_state.workflow_status is WorkflowStatus.FAILED
    assert final_state.errors[0].code == "evidence_conflict"
    assert final_state.errors[0].recoverable is True
