import pytest

from app.schemas import (
    CitationMetadata,
    KnowledgeSourceType,
    PolicyRAGResult,
    PolicyRAGStatus,
    RetrievalResult,
    RetrievalStats,
    SpecialistInput,
    SpecialistKind,
    WorkflowStatus,
)
from app.workflows.policy_rag_graph import build_policy_rag_graph


class StubPolicyRAGService:
    def __init__(self, result: PolicyRAGResult) -> None:
        self.result = result
        self.question = None
        self.conversation_context = None

    def answer(
        self,
        question: str,
        *,
        conversation_context: str = "",
    ) -> PolicyRAGResult:
        self.question = question
        self.conversation_context = conversation_context
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

    result = graph.invoke(
        SpecialistInput(
            specialist=SpecialistKind.POLICY,
            request_id="req-policy",
            session_id="session-policy",
            user_message="What does policy say?",
            conversation_context="Teacher prefers concise answers.",
        )
    )

    assert result.status is WorkflowStatus.COMPLETED
    assert result.draft is not None
    assert result.draft.title == "Policy answer draft"
    assert result.draft.content == "Use evidence cautiously [E1]."
    assert result.citations[0].source == "eylf-v2"
    assert result.citations[0].page == 21
    assert result.trace[-1].step == "policy_rag"
    assert service.question == "What does policy say?"
    assert service.conversation_context == "Teacher prefers concise answers."


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

    result = graph.invoke(
        SpecialistInput(
            specialist=SpecialistKind.POLICY,
            request_id="req-policy",
            session_id="session-policy",
            user_message="What policy?",
        )
    )

    assert result.needs_clarification is True
    assert result.clarification_question == "Which policy area should I search?"
    assert result.trace[-1].metadata["status"] == "needs_clarification"


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

    result = graph.invoke(
        SpecialistInput(
            specialist=SpecialistKind.POLICY,
            request_id="req-policy",
            session_id="session-policy",
            user_message="What policy?",
        )
    )

    assert result.status is WorkflowStatus.FAILED
    assert result.errors[0].code == "evidence_conflict"
    assert result.errors[0].recoverable is True


def test_policy_rag_graph_rejects_wrong_specialist_kind() -> None:
    graph = build_policy_rag_graph(
        StubPolicyRAGService(
            PolicyRAGResult(
                status=PolicyRAGStatus.NEEDS_CLARIFICATION,
                question="What policy?",
                clarification_question="Which policy area?",
                retrieval=empty_retrieval(),
            )
        )
    )

    with pytest.raises(ValueError, match="specialist=policy"):
        graph.invoke(
            SpecialistInput(
                specialist=SpecialistKind.PLANNING,
                request_id="req-wrong-policy",
                session_id="session-wrong-policy",
                user_message="Plan an activity.",
            )
        )
