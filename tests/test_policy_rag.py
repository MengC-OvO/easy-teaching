from typing import List

from app.schemas import (
    CitationMetadata,
    KnowledgeSourceType,
    PolicyRAGStatus,
    RetrievedKnowledgeChunk,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStats,
)
from app.services import ModelTimeoutError
from app.services import ModelResponse, PolicyRAGService
from app.services.model_types import ModelRequest


class FakePolicyRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.calls: List[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.calls.append(request)
        return self.result


class FakePolicyAnswerModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: List[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return ModelResponse(content=self.content, model="fake-policy-model")


def make_retrieved_chunk(
    chunk_id: str,
    *,
    source_id: str = "eylf-v2",
    version: str = "2.0-2022",
    page: int = 21,
) -> RetrievedKnowledgeChunk:
    return RetrievedKnowledgeChunk(
        chunk_id=chunk_id,
        content="Play-based learning provides opportunities for children.",
        citation=CitationMetadata(
            source_id=source_id,
            source_type=KnowledgeSourceType.OFFICIAL,
            title="EYLF V2.0",
            version=version,
            page=page,
        ),
        content_hash=chunk_id.ljust(64, "a")[:64],
        distance=0.2,
    )


def make_retrieval_result(chunks: List[RetrievedKnowledgeChunk]) -> RetrievalResult:
    return RetrievalResult(
        query="What does EYLF say about play?",
        chunks=chunks,
        stats=RetrievalStats(
            requested_top_k=5,
            raw_result_count=len(chunks),
            deduplicated_count=len(chunks),
            returned_count=len(chunks),
        ),
    )


def test_policy_rag_answers_with_evidence_and_citations() -> None:
    retriever = FakePolicyRetriever(
        make_retrieval_result([make_retrieved_chunk("chunk-1")])
    )
    service = PolicyRAGService(retriever=retriever)

    result = service.answer("What does EYLF say about play?")

    assert result.status is PolicyRAGStatus.ANSWERED
    assert result.answer is not None
    assert "Draft policy answer based only on retrieved evidence" in result.answer
    assert "[E1]" in result.answer
    assert result.evidence[0].evidence_id == "E1"
    assert result.citations[0].source_id == "eylf-v2"
    assert retriever.calls[0].query == "What does EYLF say about play?"


def test_policy_rag_uses_model_provider_for_grounded_answer() -> None:
    retriever = FakePolicyRetriever(
        make_retrieval_result([make_retrieved_chunk("chunk-1")])
    )
    model_provider = FakePolicyAnswerModel(
        "Play-based learning is described as a context and process for learning [E1]."
    )
    service = PolicyRAGService(
        retriever=retriever,
        model_provider=model_provider,
    )

    result = service.answer("What does EYLF say about play?")

    assert result.status is PolicyRAGStatus.ANSWERED
    assert result.answer == (
        "Play-based learning is described as a context and process for learning [E1]."
    )
    assert len(model_provider.calls) == 1
    messages = model_provider.calls[0].messages
    assert messages[0].role.value == "system"
    assert "Answer only from the provided evidence" in messages[0].content
    assert "[E1]" in messages[1].content
    assert "Play-based learning provides opportunities" in messages[1].content


def test_policy_rag_falls_back_to_local_evidence_when_model_fails() -> None:
    class FailingPolicyAnswerModel:
        def generate(self, request):
            raise ModelTimeoutError("model timed out")

    retriever = FakePolicyRetriever(
        make_retrieval_result([make_retrieved_chunk("chunk-1")])
    )
    service = PolicyRAGService(
        retriever=retriever,
        model_provider=FailingPolicyAnswerModel(),
    )

    result = service.answer("What does EYLF say about play?")

    assert result.status is PolicyRAGStatus.ANSWERED
    assert "Draft policy answer based only on retrieved evidence" in result.answer
    assert "[E1]" in result.answer
    assert result.generation_fallback is True
    assert result.generation_error_code == "timeout"


def test_policy_rag_clarifies_when_retrieval_is_empty() -> None:
    retriever = FakePolicyRetriever(make_retrieval_result([]))
    model_provider = FakePolicyAnswerModel("Should not be called")
    service = PolicyRAGService(retriever=retriever, model_provider=model_provider)

    result = service.answer("What does the policy say?")

    assert result.status is PolicyRAGStatus.NEEDS_CLARIFICATION
    assert result.clarification_question is not None
    assert result.answer is None
    assert result.evidence == []
    assert model_provider.calls == []


def test_policy_rag_detects_conflicting_versions_for_same_source() -> None:
    retriever = FakePolicyRetriever(
        make_retrieval_result(
            [
                make_retrieved_chunk("chunk-1", version="2.0-2022"),
                make_retrieved_chunk("chunk-2", version="1.0-2009", page=22),
            ]
        )
    )
    model_provider = FakePolicyAnswerModel("Should not be called")
    service = PolicyRAGService(retriever=retriever, model_provider=model_provider)

    result = service.answer("What does EYLF say about play?")

    assert result.status is PolicyRAGStatus.EVIDENCE_CONFLICT
    assert result.refusal_reason == "Retrieved evidence contains multiple versions for the same source."
    assert len(result.evidence) == 2
    assert len(result.citations) == 2
    assert model_provider.calls == []
