import re
from typing import List, Optional, Protocol, Type

from pydantic import BaseModel, Field

from app.schemas import (
    RetrievalMode,
    RetrievalRequest,
    RerankerMode,
    RiskLevel,
)
from app.services import (
    ChatCompletionsModelProvider,
    KnowledgeRetriever,
    ModelMessage,
    ModelResponse,
    ModelRole,
)
from app.tools.controlled_tools.retrieve_risk_guidance import (
    KnowledgeEvidenceItem,
    KnowledgeRetrieverProtocol,
    retrieval_to_evidence,
)
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolPermission,
    ToolResult,
)


class AlignToEylfOutcomesInput(BaseModel):
    activity_text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class EylfOutcomeAlignment(BaseModel):
    outcome: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: List[str]


class AlignToEylfOutcomesOutput(BaseModel):
    alignments: List[EylfOutcomeAlignment]
    evidence: List[KnowledgeEvidenceItem]
    mode: RetrievalMode
    reranker: RerankerMode


class EylfAlignmentModelProvider(Protocol):
    def generate_structured(
        self,
        *,
        messages: List[ModelMessage],
        response_model: Type[AlignToEylfOutcomesOutput],
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...


def build_align_to_eylf_outcomes_tool(
    retriever: Optional[KnowledgeRetrieverProtocol] = None,
    model_provider: Optional[EylfAlignmentModelProvider] = None,
) -> ToolDefinition:
    resolved_retriever = retriever
    resolved_model_provider = model_provider

    def handler(input_data: BaseModel) -> ToolResult:
        nonlocal resolved_retriever, resolved_model_provider
        data = AlignToEylfOutcomesInput.model_validate(input_data)
        if resolved_retriever is None:
            resolved_retriever = KnowledgeRetriever()
        if resolved_model_provider is None:
            resolved_model_provider = ChatCompletionsModelProvider()
        retrieval = resolved_retriever.retrieve(
            RetrievalRequest(
                query=eylf_alignment_query(data.activity_text),
                top_k=data.top_k,
                mode=RetrievalMode.HYBRID,
                reranker=RerankerMode.LEXICAL,
            )
        )
        evidence = retrieval_to_evidence(retrieval)
        model_result = resolved_model_provider.generate_structured(
            messages=build_eylf_alignment_messages(data.activity_text, evidence),
            response_model=AlignToEylfOutcomesOutput,
            temperature=0.0,
        )
        if not isinstance(model_result.structured, AlignToEylfOutcomesOutput):
            raise TypeError("EYLF alignment provider returned an unexpected result")
        alignments = [
            alignment.model_dump(mode="json")
            for alignment in model_result.structured.alignments
        ]
        return ToolResult.ok(
            data={
                "alignments": alignments,
                "evidence": evidence,
                "mode": retrieval.stats.mode.value,
                "reranker": retrieval.stats.reranker.value,
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="align_to_eylf_outcomes",
        description=(
            "Map an activity draft to likely EYLF learning outcomes using "
            "curriculum framework guidance."
        ),
        category=ToolCategory.CURRICULUM,
        input_model=AlignToEylfOutcomesInput,
        output_model=AlignToEylfOutcomesOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )


def eylf_alignment_query(activity_text: str) -> str:
    keywords = " ".join(token for token in re.findall(r"[A-Za-z]{4,}", activity_text)[:20])
    return (
        "EYLF curriculum framework learning outcomes principles practices play based learning "
        f"{keywords}"
    ).strip()


def build_eylf_alignment_messages(
    activity_text: str,
    evidence: List[dict[str, object]],
) -> List[ModelMessage]:
    evidence_text = "\n\n".join(
        f"[{item['evidence_id']}] {item['content']}" for item in evidence
    )
    return [
        ModelMessage(
            role=ModelRole.SYSTEM,
            content=(
                "You align early childhood activity drafts to EYLF learning outcomes. "
                "Use only the supplied evidence. Return structured JSON matching the "
                "requested schema. Each alignment must cite evidence_ids that support it. "
                "Do not invent EYLF content that is not supported by the evidence."
            ),
        ),
        ModelMessage(
            role=ModelRole.USER,
            content=(
                f"Activity draft:\n{activity_text}\n\n"
                f"Retrieved EYLF evidence:\n{evidence_text}\n\n"
                "Select the most relevant EYLF outcomes, explain why each fits, "
                "assign confidence from 0 to 1, and include supporting evidence_ids."
            ),
        ),
    ]
