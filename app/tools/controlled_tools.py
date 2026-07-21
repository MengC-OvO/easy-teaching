import re
from typing import Dict, List, Optional, Protocol

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


class RetrievePolicyEvidenceInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    source_type: Optional[KnowledgeSourceType] = None


class PolicyEvidenceItem(BaseModel):
    evidence_id: str
    content: str
    citation: CitationMetadata
    distance: float = Field(ge=0)
    content_hash: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class RetrievePolicyEvidenceOutput(BaseModel):
    evidence: List[PolicyEvidenceItem]
    mode: RetrievalMode
    reranker: RerankerMode
    returned_count: int = Field(ge=0)


class CheckActivitySafetyInput(BaseModel):
    activity_text: str = Field(min_length=1)
    age_group: Optional[str] = None
    class_size: Optional[int] = Field(default=None, ge=1)


class SafetyCheckItem(BaseModel):
    code: str
    severity: str
    message: str
    suggestion: str


class CheckActivitySafetyOutput(BaseModel):
    status: str
    issues: List[SafetyCheckItem]
    checked_risk_terms: List[str]


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
    evidence: List[PolicyEvidenceItem]
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


def build_retrieve_policy_evidence_tool(
    retriever: Optional[PolicyRetrieverProtocol] = None,
) -> ToolDefinition:
    resolved_retriever = retriever

    def handler(input_data: BaseModel) -> ToolResult:
        nonlocal resolved_retriever
        data = RetrievePolicyEvidenceInput.model_validate(input_data)
        if resolved_retriever is None:
            resolved_retriever = KnowledgeRetriever()
        filters = RetrievalFilters()
        if data.source_type is not None:
            filters.source_types = [data.source_type]
        retrieval = resolved_retriever.retrieve(
            RetrievalRequest(
                query=data.query,
                top_k=data.top_k,
                filters=filters,
                mode=RetrievalMode.BM25,
                reranker=RerankerMode.LEXICAL,
            )
        )
        return ToolResult.ok(
            data={
                "evidence": _retrieval_to_evidence(retrieval),
                "mode": retrieval.stats.mode.value,
                "reranker": retrieval.stats.reranker.value,
                "returned_count": retrieval.stats.returned_count,
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="retrieve_policy_evidence",
        description="Retrieve citable local policy evidence from the knowledge base.",
        category=ToolCategory.POLICY,
        input_model=RetrievePolicyEvidenceInput,
        output_model=RetrievePolicyEvidenceOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )


def build_check_activity_safety_tool() -> ToolDefinition:
    def handler(input_data: BaseModel) -> ToolResult:
        data = CheckActivitySafetyInput.model_validate(input_data)
        issues = _check_activity_safety(
            data.activity_text,
            age_group=data.age_group,
            class_size=data.class_size,
        )
        status = "needs_revision" if issues else "passed"
        return ToolResult.ok(
            data={
                "status": status,
                "issues": issues,
                "checked_risk_terms": sorted(SAFETY_RISK_RULES),
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="check_activity_safety",
        description="Check an activity draft for common early childhood safety risks.",
        category=ToolCategory.SAFETY,
        input_model=CheckActivitySafetyInput,
        output_model=CheckActivitySafetyOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )


def build_align_to_eylf_outcomes_tool(
    retriever: Optional[PolicyRetrieverProtocol] = None,
) -> ToolDefinition:
    resolved_retriever = retriever

    def handler(input_data: BaseModel) -> ToolResult:
        nonlocal resolved_retriever
        data = AlignToEylfOutcomesInput.model_validate(input_data)
        if resolved_retriever is None:
            resolved_retriever = KnowledgeRetriever()
        retrieval = resolved_retriever.retrieve(
            RetrievalRequest(
                query=_eylf_alignment_query(data.activity_text),
                top_k=data.top_k,
                mode=RetrievalMode.BM25,
                reranker=RerankerMode.LEXICAL,
            )
        )
        evidence = _retrieval_to_evidence(retrieval)
        alignments = _infer_eylf_alignments(data.activity_text, evidence)
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
        description="Map an activity draft to likely EYLF outcomes using retrieved evidence.",
        category=ToolCategory.CURRICULUM,
        input_model=AlignToEylfOutcomesInput,
        output_model=AlignToEylfOutcomesOutput,
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
        build_retrieve_policy_evidence_tool(policy_retriever),
        build_check_activity_safety_tool(),
        build_align_to_eylf_outcomes_tool(policy_retriever),
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


SAFETY_RISK_RULES = {
    "water": (
        "high",
        "Water-based activities need explicit supervision, boundaries, and emergency planning.",
        "Add active supervision, safe depth/boundaries, and a dry alternative.",
    ),
    "food": (
        "medium",
        "Food activities need allergy and choking checks.",
        "Check allergies, use age-appropriate textures, and avoid shared utensils.",
    ),
    "allergy": (
        "medium",
        "The draft mentions allergies or possible allergens.",
        "Add allergy checks and substitution options before the activity runs.",
    ),
    "scissors": (
        "medium",
        "Sharp tools need age-appropriate supervision.",
        "Specify child-safe scissors and close educator supervision.",
    ),
    "heat": (
        "high",
        "Heat, cooking, or hot materials need stronger controls.",
        "Keep children away from hot surfaces and use educator-only handling.",
    ),
    "outdoor": (
        "medium",
        "Outdoor activities need weather, boundary, and supervision planning.",
        "Add sun protection, clear boundaries, head counts, and a wet-weather option.",
    ),
}


EYLF_OUTCOME_RULES = {
    "Outcome 1": {
        "terms": {"identity", "belonging", "agency", "confidence", "relationships"},
        "reason": "The activity appears to support identity, agency, belonging, or secure relationships.",
    },
    "Outcome 2": {
        "terms": {"community", "environment", "nature", "outdoor", "sustainability", "world"},
        "reason": "The activity connects children with community, environment, or their wider world.",
    },
    "Outcome 3": {
        "terms": {"wellbeing", "movement", "sensory", "body", "health", "physical"},
        "reason": "The activity includes wellbeing, sensory, movement, or physical development elements.",
    },
    "Outcome 4": {
        "terms": {"play", "explore", "experiment", "problem", "curiosity", "thinking", "investigate"},
        "reason": "The activity encourages play, inquiry, curiosity, problem solving, or confident learning.",
    },
    "Outcome 5": {
        "terms": {"language", "describe", "story", "symbols", "communication", "draw", "write"},
        "reason": "The activity gives children opportunities to communicate, describe, represent, or use language.",
    },
}


def _retrieval_to_evidence(retrieval: RetrievalResult) -> List[Dict[str, object]]:
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


def _check_activity_safety(
    activity_text: str,
    *,
    age_group: Optional[str],
    class_size: Optional[int],
) -> List[Dict[str, str]]:
    normalized = activity_text.lower()
    issues: List[Dict[str, str]] = []
    for term, (severity, message, suggestion) in SAFETY_RISK_RULES.items():
        if term in normalized:
            issues.append(
                {
                    "code": f"activity_contains_{term}",
                    "severity": severity,
                    "message": message,
                    "suggestion": suggestion,
                }
            )

    if class_size is not None and class_size >= 20:
        issues.append(
            {
                "code": "large_group_supervision",
                "severity": "medium",
                "message": "Large-group activities need explicit supervision planning.",
                "suggestion": "Split children into small groups or name educator supervision roles.",
            }
        )

    if age_group and re.search(r"\b(0-2|1-2|toddlers?)\b", age_group.lower()):
        issues.append(
            {
                "code": "younger_children_adaptation",
                "severity": "medium",
                "message": "Activities for younger children need choking, mouthing, and mobility adjustments.",
                "suggestion": "Use larger materials, avoid small loose parts, and add close supervision.",
            }
        )

    return issues


def _eylf_alignment_query(activity_text: str) -> str:
    keywords = " ".join(token for token in re.findall(r"[A-Za-z]{4,}", activity_text)[:20])
    return (
        "EYLF outcomes play based learning children learning outcome "
        f"{keywords}"
    ).strip()


def _infer_eylf_alignments(
    activity_text: str,
    evidence: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    searchable_text = " ".join(
        [
            activity_text.lower(),
            " ".join(str(item["content"]).lower() for item in evidence),
        ]
    )
    alignments: List[Dict[str, object]] = []
    evidence_ids = [str(item["evidence_id"]) for item in evidence[:3]]
    for outcome, rule in EYLF_OUTCOME_RULES.items():
        terms = rule["terms"]
        matched_terms = [term for term in terms if term in searchable_text]
        if not matched_terms:
            continue
        confidence = min(0.95, 0.45 + 0.1 * len(matched_terms))
        alignments.append(
            {
                "outcome": outcome,
                "reason": rule["reason"],
                "confidence": round(confidence, 2),
                "evidence_ids": evidence_ids,
            }
        )

    return alignments[:3]
