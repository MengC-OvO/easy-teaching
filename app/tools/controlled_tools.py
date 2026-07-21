from typing import List, Optional, Protocol

from pydantic import BaseModel, Field

from app.schemas import (
    CitationMetadata,
    RetrievalMode,
    RetrievalRequest,
    RetrievalResult,
    RerankerMode,
    RiskLevel,
)
from app.services import EduFlowStore, KnowledgeRetriever
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolErrorCode,
    ToolPermission,
    ToolResult,
)
from app.tools.registry import ToolRegistry


class GetClassProfileInput(BaseModel):
    class_id: str = Field(min_length=1)


class GetClassProfileOutput(BaseModel):
    class_id: str
    name: str
    age_group: str
    child_count: int
    interests: List[str]
    safety_notes: List[str]


class SearchPolicyIndexInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class PolicyIndexItem(BaseModel):
    policy_id: str
    title: str
    source: str
    section: str
    summary: str
    citation: CitationMetadata
    distance: float = Field(ge=0)
    content_hash: str


class SearchPolicyIndexOutput(BaseModel):
    results: List[PolicyIndexItem]
    mode: RetrievalMode
    reranker: RerankerMode


class PolicyRetrieverProtocol(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        ...


class SaveDraftInput(BaseModel):
    draft_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    draft_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SaveDraftOutput(BaseModel):
    draft_id: str
    draft_type: str
    title: str
    status: str


def build_get_class_profile_tool(store: EduFlowStore) -> ToolDefinition:
    def handler(input_data: BaseModel) -> ToolResult:
        data = GetClassProfileInput.model_validate(input_data)
        profile = store.get_class_profile(data.class_id)
        if profile is None:
            return ToolResult.fail(
                code=ToolErrorCode.EXECUTION_ERROR,
                message=f"Class profile not found: {data.class_id}",
                risk_level=RiskLevel.L0_READ_ONLY,
                recoverable=True,
                details={"class_id": data.class_id},
            )
        return ToolResult.ok(data=profile, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="get_class_profile",
        description="Read a synthetic class profile by class id.",
        category=ToolCategory.CLASS_PROFILE,
        input_model=GetClassProfileInput,
        output_model=GetClassProfileOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )


def build_search_policy_index_tool(
    retriever: Optional[PolicyRetrieverProtocol] = None,
) -> ToolDefinition:
    resolved_retriever = retriever

    def handler(input_data: BaseModel) -> ToolResult:
        nonlocal resolved_retriever
        data = SearchPolicyIndexInput.model_validate(input_data)
        if resolved_retriever is None:
            resolved_retriever = KnowledgeRetriever()
        retrieval = resolved_retriever.retrieve(
            RetrievalRequest(
                query=data.query,
                top_k=data.top_k,
                mode=RetrievalMode.BM25,
                reranker=RerankerMode.LEXICAL,
            )
        )
        results = [
            {
                "policy_id": chunk.chunk_id,
                "title": chunk.citation.title,
                "source": chunk.citation.source_id,
                "section": chunk.citation.section or f"page {chunk.citation.page}",
                "summary": chunk.content,
                "citation": chunk.citation.model_dump(mode="json"),
                "distance": chunk.distance,
                "content_hash": chunk.content_hash,
            }
            for chunk in retrieval.chunks
        ]
        return ToolResult.ok(
            data={
                "results": results,
                "mode": retrieval.stats.mode.value,
                "reranker": retrieval.stats.reranker.value,
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="search_policy_index",
        description="Search the local knowledge retriever for policy evidence.",
        category=ToolCategory.POLICY,
        input_model=SearchPolicyIndexInput,
        output_model=SearchPolicyIndexOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )


def build_save_draft_tool(store: EduFlowStore) -> ToolDefinition:
    def handler(input_data: BaseModel) -> ToolResult:
        data = SaveDraftInput.model_validate(input_data)
        saved = store.save_draft(
            draft_id=data.draft_id,
            draft_type=data.draft_type,
            title=data.title,
            content=data.content,
            idempotency_key=data.idempotency_key,
        )
        return ToolResult.ok(data=saved, risk_level=RiskLevel.L2_CONTROLLED_WRITE)

    return ToolDefinition(
        name="save_draft",
        description="Save a draft record after teacher approval.",
        category=ToolCategory.DRAFT,
        input_model=SaveDraftInput,
        output_model=SaveDraftOutput,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        handler=handler,
    )


def build_default_tool_definitions(
    store: EduFlowStore,
    *,
    policy_retriever: Optional[PolicyRetrieverProtocol] = None,
) -> List[ToolDefinition]:
    return [
        build_get_class_profile_tool(store),
        build_search_policy_index_tool(policy_retriever),
        build_save_draft_tool(store),
    ]


def build_default_tool_registry(
    store: EduFlowStore,
    *,
    policy_retriever: Optional[PolicyRetrieverProtocol] = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_default_tool_definitions(
        store,
        policy_retriever=policy_retriever,
    ):
        registry.register(tool)
    return registry
