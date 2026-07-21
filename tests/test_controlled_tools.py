from app.schemas import (
    CitationMetadata,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    RetrievalResult,
    RetrievalStats,
    RetrievalMode,
    RerankerMode,
    RiskLevel,
)
from app.services import EduFlowStore
from app.tools import ToolErrorCode, build_default_tool_registry


def make_store(tmp_path) -> EduFlowStore:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'eduflow-test.sqlite3'}")
    store.initialize()
    return store


class StubPolicyRetriever:
    def __init__(self) -> None:
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        citation = CitationMetadata(
            source_id="eylf-v2",
            source_type=KnowledgeSourceType.OFFICIAL,
            title="EYLF V2.0",
            version="2.0-2022",
            section="Learning through play",
            page=21,
            uri="https://example.test/eylf.pdf",
        )
        return RetrievalResult(
            query=request.query,
            chunks=[
                RetrievedKnowledgeChunk(
                    chunk_id="chunk-001",
                    content="Play-based learning provides opportunities for children.",
                    citation=citation,
                    content_hash="a" * 64,
                    distance=0.25,
                    metadata={"bm25_score": "3.140000"},
                )
            ],
            stats=RetrievalStats(
                requested_top_k=request.top_k,
                mode=request.mode,
                reranker=request.reranker,
                raw_result_count=1,
                bm25_result_count=1,
                deduplicated_count=1,
                returned_count=1,
                reranked=True,
            ),
        )


def test_default_tool_registry_registers_controlled_tools(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    assert [tool.name for tool in registry.list_tools()] == [
        "get_class_profile",
        "retrieve_policy_evidence",
        "check_activity_safety",
        "align_to_eylf_outcomes",
        "save_draft",
    ]


def test_get_class_profile_tool_reads_synthetic_data(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = registry.execute("get_class_profile", {"class_id": "kangaroo-room"})

    assert result.success is True
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.data["name"] == "Kangaroo Room"
    assert "synthetic data only" in result.data["safety_notes"]
    assert result.trace is not None
    assert result.trace.tool_name == "get_class_profile"


def test_retrieve_policy_evidence_tool_returns_citable_chunks(tmp_path) -> None:
    retriever = StubPolicyRetriever()
    registry = build_default_tool_registry(
        make_store(tmp_path),
        policy_retriever=retriever,
    )

    result = registry.execute(
        "retrieve_policy_evidence",
        {"query": "play based learning", "top_k": 4, "source_type": "official"},
    )

    assert result.success is True
    assert result.data["returned_count"] == 1
    assert result.data["evidence"][0]["evidence_id"] == "E1"
    assert result.data["evidence"][0]["citation"]["source_id"] == "eylf-v2"
    assert retriever.requests[0].query == "play based learning"
    assert retriever.requests[0].top_k == 4
    assert retriever.requests[0].filters.source_types == [KnowledgeSourceType.OFFICIAL]


def test_check_activity_safety_tool_flags_common_risks(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "check_activity_safety",
        {
            "activity_text": "Outdoor water play with food dye and scissors.",
            "age_group": "3-5",
            "class_size": 22,
        },
    )

    assert result.success is True
    assert result.data["status"] == "needs_revision"
    assert {issue["code"] for issue in result.data["issues"]} >= {
        "activity_contains_water",
        "activity_contains_food",
        "activity_contains_scissors",
        "activity_contains_outdoor",
        "large_group_supervision",
    }


def test_align_to_eylf_outcomes_tool_uses_retrieved_evidence(tmp_path) -> None:
    retriever = StubPolicyRetriever()
    registry = build_default_tool_registry(
        make_store(tmp_path),
        policy_retriever=retriever,
    )

    result = registry.execute(
        "align_to_eylf_outcomes",
        {
            "activity_text": (
                "Children explore outdoor natural materials through play, "
                "describe textures, and solve problems together."
            ),
            "top_k": 3,
        },
    )

    assert result.success is True
    assert result.data["evidence"][0]["evidence_id"] == "E1"
    assert [item["outcome"] for item in result.data["alignments"]] == [
        "Outcome 2",
        "Outcome 4",
        "Outcome 5",
    ]
    assert result.data["alignments"][0]["evidence_ids"] == ["E1"]
    assert "EYLF outcomes" in retriever.requests[0].query


def test_save_draft_tool_requires_approval(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert result.risk_level is RiskLevel.L2_CONTROLLED_WRITE


def test_save_draft_tool_writes_after_approval(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
        approved=True,
    )

    assert result.success is True
    assert result.data == {
        "draft_id": "draft-001",
        "draft_type": "activity_plan",
        "title": "Outdoor sensory walk",
        "status": "draft",
    }


def test_save_draft_tool_requires_idempotency_key(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
        approved=True,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.VALIDATION_ERROR
    assert result.error.details["errors"][0]["loc"] == ("idempotency_key",)


def test_save_draft_tool_reuses_existing_result_for_same_idempotency_key(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    first = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-001",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Outdoor sensory walk",
            "content": "Synthetic draft content.",
        },
        approved=True,
    )
    second = registry.execute(
        "save_draft",
        {
            "draft_id": "draft-002",
            "idempotency_key": "request-001:save-draft",
            "draft_type": "activity_plan",
            "title": "Should not replace original",
            "content": "Different synthetic content.",
        },
        approved=True,
    )

    assert first.success is True
    assert second.success is True
    assert second.data == first.data


def test_get_class_profile_tool_reports_missing_profile(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = registry.execute("get_class_profile", {"class_id": "missing-room"})

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_ERROR
    assert result.error.details == {"class_id": "missing-room"}
