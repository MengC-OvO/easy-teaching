from typing import List, Optional, Protocol, Set, Tuple

from app.schemas import (
    PolicyEvidence,
    PolicyRAGResult,
    PolicyRAGStatus,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
    RerankerMode,
)
from app.services.knowledge_retriever import KnowledgeRetriever
from app.services.model_errors import ModelProviderError
from app.services.model_types import ModelMessage, ModelRequest, ModelResponse, ModelRole


class PolicyRetriever(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        ...


class PolicyAnswerModel(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse:
        ...


class PolicyRAGService:
    def __init__(
        self,
        *,
        retriever: Optional[PolicyRetriever] = None,
        model_provider: Optional[PolicyAnswerModel] = None,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        reranker: RerankerMode = RerankerMode.CROSS_ENCODER,
    ) -> None:
        self.retriever = retriever or KnowledgeRetriever()
        self.model_provider = model_provider
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.reranker = reranker

    def answer(
        self,
        question: str,
        *,
        conversation_context: str = "",
    ) -> PolicyRAGResult:
        retrieval = self.retriever.retrieve(
            RetrievalRequest(
                query=question,
                top_k=self.top_k,
                mode=self.retrieval_mode,
                reranker=self.reranker,
            )
        )

        if retrieval.stats.returned_count == 0:
            return PolicyRAGResult(
                status=PolicyRAGStatus.NEEDS_CLARIFICATION,
                question=question,
                clarification_question=(
                    "I could not find enough policy evidence. "
                    "Can you specify the policy area, source, or scenario?"
                ),
                retrieval=retrieval,
            )

        evidence = self._build_evidence(retrieval)
        conflict = self._detect_conflict(evidence)
        if conflict:
            return PolicyRAGResult(
                status=PolicyRAGStatus.EVIDENCE_CONFLICT,
                question=question,
                evidence=evidence,
                citations=[item.citation for item in evidence],
                refusal_reason=conflict,
                retrieval=retrieval,
            )

        generation_fallback = False
        generation_error_code = None
        try:
            answer = self._generate_grounded_answer(
                question,
                evidence,
                conversation_context=conversation_context,
            )
        except ModelProviderError as error:
            answer = self._build_evidence_answer(question, evidence)
            generation_fallback = True
            generation_error_code = error.code.value

        return PolicyRAGResult(
            status=PolicyRAGStatus.ANSWERED,
            question=question,
            answer=answer,
            evidence=evidence,
            citations=[item.citation for item in evidence],
            generation_fallback=generation_fallback,
            generation_error_code=generation_error_code,
            retrieval=retrieval,
        )

    def _build_evidence(self, retrieval: RetrievalResult) -> List[PolicyEvidence]:
        return [
            PolicyEvidence.from_retrieved_chunk(chunk, index=index)
            for index, chunk in enumerate(retrieval.chunks, start=1)
        ]

    def _detect_conflict(self, evidence: List[PolicyEvidence]) -> str:
        source_versions: Set[Tuple[str, str]] = {
            (item.citation.source_id, item.citation.version)
            for item in evidence
        }
        source_ids = {source_id for source_id, _ in source_versions}
        if len(source_versions) > len(source_ids):
            return "Retrieved evidence contains multiple versions for the same source."
        return ""

    def _generate_grounded_answer(
        self,
        question: str,
        evidence: List[PolicyEvidence],
        *,
        conversation_context: str = "",
    ) -> str:
        if self.model_provider is None:
            return self._build_evidence_answer(question, evidence)

        response = self.model_provider.generate(
            ModelRequest(
                messages=[
                    ModelMessage(
                        role=ModelRole.SYSTEM,
                        content=self._system_prompt(),
                    ),
                    ModelMessage(
                        role=ModelRole.USER,
                        content=self._user_prompt(
                            question,
                            evidence,
                            conversation_context=conversation_context,
                        ),
                    ),
                ],
                temperature=0.0,
            )
        )
        return response.content.strip()

    def _system_prompt(self) -> str:
        return (
            "You are EduFlow AU, a cautious teacher assistant for Australian "
            "early childhood education policy questions. Answer only from the "
            "provided evidence. Cite evidence IDs like [E1] for every substantive "
            "claim. If the evidence is insufficient, say what is missing. Do not "
            "provide legal advice, medical advice, diagnoses, or compliance "
            "conclusions."
        )

    def _user_prompt(
        self,
        question: str,
        evidence: List[PolicyEvidence],
        *,
        conversation_context: str = "",
    ) -> str:
        evidence_blocks = []
        for item in evidence:
            evidence_blocks.append(
                "\n".join(
                    [
                        f"[{item.evidence_id}]",
                        f"Source: {self._citation_location(item)}",
                        f"Text: {item.content}",
                    ]
                )
            )
        joined_evidence = "\n\n".join(evidence_blocks)
        context_block = (
            f"Conversation context (background only, not evidence):\n{conversation_context}\n\n"
            if conversation_context
            else ""
        )
        return (
            f"Question: {question}\n\n"
            f"{context_block}"
            "Evidence:\n"
            f"{joined_evidence}\n\n"
            "Write a concise policy answer for a teacher. Use citations like "
            "[E1] after claims. Keep the answer cautious and draft-like."
        )

    def _build_evidence_answer(
        self,
        question: str,
        evidence: List[PolicyEvidence],
    ) -> str:
        evidence_lines = []
        for item in evidence:
            location = self._citation_location(item)
            evidence_lines.append(
                f"[{item.evidence_id}] {item.content} ({location})"
            )
        joined_evidence = "\n".join(evidence_lines)
        return (
            "Draft policy answer based only on retrieved evidence.\n\n"
            f"Question: {question}\n\n"
            "Evidence:\n"
            f"{joined_evidence}\n\n"
            "Use the cited evidence above to make a cautious policy response. "
            "Do not treat this as legal advice."
        )

    def _citation_location(self, evidence: PolicyEvidence) -> str:
        citation = evidence.citation
        parts = [citation.title, f"version {citation.version}"]
        if citation.section:
            parts.append(f"section {citation.section}")
        if citation.page:
            parts.append(f"page {citation.page}")
        return ", ".join(parts)
