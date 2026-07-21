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
        "search_policy_index",
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


def test_search_policy_index_tool_uses_knowledge_retriever(tmp_path) -> None:
    retriever = StubPolicyRetriever()
    registry = build_default_tool_registry(
        make_store(tmp_path),
        policy_retriever=retriever,
    )

    result = registry.execute("search_policy_index", {"query": "play", "top_k": 2})

    assert result.success is True
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.data["mode"] == RetrievalMode.BM25.value
    assert result.data["reranker"] == RerankerMode.LEXICAL.value
    assert result.data["results"][0]["policy_id"] == "chunk-001"
    assert result.data["results"][0]["source"] == "eylf-v2"
    assert result.data["results"][0]["citation"]["page"] == 21
    assert retriever.requests[0].query == "play"
    assert retriever.requests[0].top_k == 2
    assert retriever.requests[0].mode is RetrievalMode.BM25
    assert retriever.requests[0].reranker is RerankerMode.LEXICAL


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
