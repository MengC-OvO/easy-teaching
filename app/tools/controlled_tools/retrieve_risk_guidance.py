from typing import Dict, List, Optional, Protocol, Type

from pydantic import BaseModel, Field

from app.schemas import (
    CitationMetadata,
    KnowledgeSourceType,
    RetrievalFilters,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
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
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolPermission,
    ToolResult,
)


class RetrieveRiskGuidanceInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    source_type: Optional[KnowledgeSourceType] = None


class KnowledgeEvidenceItem(BaseModel):
    evidence_id: str
    content: str
    citation: CitationMetadata
    distance: float = Field(ge=0)
    content_hash: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class RetrieveRiskGuidanceOutput(BaseModel):
    guidance_summary: str
    risk_level: str
    required_controls: List[str]
    evidence_ids: List[str]
    evidence: List[KnowledgeEvidenceItem]
    mode: RetrievalMode
    reranker: RerankerMode
    returned_count: int = Field(ge=0)


class KnowledgeRetrieverProtocol(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        ...


class RiskGuidanceModelProvider(Protocol):
    def generate_structured(
        self,
        *,
        messages: List[ModelMessage],
        response_model: Type[RetrieveRiskGuidanceOutput],
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...


def build_retrieve_risk_guidance_tool(
    retriever: Optional[KnowledgeRetrieverProtocol] = None,
    model_provider: Optional[RiskGuidanceModelProvider] = None,
) -> ToolDefinition:
    resolved_retriever = retriever
    resolved_model_provider = model_provider

    def handler(input_data: BaseModel) -> ToolResult:
        nonlocal resolved_retriever, resolved_model_provider
        data = RetrieveRiskGuidanceInput.model_validate(input_data)
        if resolved_retriever is None:
            resolved_retriever = KnowledgeRetriever()
        if resolved_model_provider is None:
            resolved_model_provider = ChatCompletionsModelProvider()
        filters = RetrievalFilters()
        if data.source_type is not None:
            filters.source_types = [data.source_type]
        retrieval = resolved_retriever.retrieve(
            RetrievalRequest(
                query=risk_guidance_query(data.query),
                top_k=data.top_k,
                filters=filters,
                mode=RetrievalMode.HYBRID,
                reranker=RerankerMode.LEXICAL,
            )
        )
        evidence = retrieval_to_evidence(retrieval)
        model_result = resolved_model_provider.generate_structured(
            messages=build_risk_guidance_messages(data.query, evidence),
            response_model=RetrieveRiskGuidanceOutput,
            temperature=0.0,
        )
        if not isinstance(model_result.structured, RetrieveRiskGuidanceOutput):
            raise TypeError("Risk guidance provider returned an unexpected result")
        return ToolResult.ok(
            data={
                "guidance_summary": model_result.structured.guidance_summary,
                "risk_level": model_result.structured.risk_level,
                "required_controls": model_result.structured.required_controls,
                "evidence_ids": model_result.structured.evidence_ids,
                "evidence": evidence,
                "mode": retrieval.stats.mode.value,
                "reranker": retrieval.stats.reranker.value,
                "returned_count": retrieval.stats.returned_count,
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="retrieve_risk_guidance",
        description=(
            "Retrieve local risk, safety, supervision, restriction, or regulatory "
            "guidance for an activity."
        ),
        category=ToolCategory.POLICY,
        input_model=RetrieveRiskGuidanceInput,
        output_model=RetrieveRiskGuidanceOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )


def retrieval_to_evidence(retrieval: RetrievalResult) -> List[Dict[str, object]]:
    return [
        {
            "evidence_id": f"E{index}",
            "content": chunk.content,
            "citation": chunk.citation.model_dump(mode="json"),
            "distance": chunk.distance,
            "content_hash": chunk.content_hash,
            "metadata": chunk.metadata,
        }
        for index, chunk in enumerate(retrieval.chunks, start=1)
    ]


def risk_guidance_query(query: str) -> str:
    return (
        "NQS NQF centre policy safety supervision risk restriction requirement "
        f"{query}"
    ).strip()


def build_risk_guidance_messages(
    query: str,
    evidence: List[Dict[str, object]],
) -> List[ModelMessage]:
    evidence_text = "\n\n".join(
        f"[{item['evidence_id']}] {item['content']}" for item in evidence
    )
    return [
        ModelMessage(
            role=ModelRole.SYSTEM,
            content=(
                "You summarize early childhood activity risk, safety, supervision, "
                "restriction, and regulatory guidance. Use only the supplied evidence. "
                "Return structured JSON matching the requested schema. risk_level "
                "must be one of low, medium, high, or unclear. required_controls "
                "must be practical controls supported by evidence_ids. Do not provide "
                "legal conclusions or invent requirements."
            ),
        ),
        ModelMessage(
            role=ModelRole.USER,
            content=(
                f"Activity/risk query:\n{query}\n\n"
                f"Retrieved risk guidance evidence:\n{evidence_text}\n\n"
                "Summarize the risk guidance, choose a risk_level, list required_controls, "
                "and include supporting evidence_ids."
            ),
        ),
    ]
