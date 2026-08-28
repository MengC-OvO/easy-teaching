import asyncio

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
from app.services import ModelResponse
from evals.in_memory_store import InMemoryEvalStore
from app.tools import (
    QueryRewriteOutput,
    ToolErrorCode,
    ToolExecutionContext,
    build_default_tool_registry,
)


def make_store(_tmp_path) -> InMemoryEvalStore:
    return InMemoryEvalStore()


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


class StubQueryRewriter:
    def __init__(self) -> None:
        self.messages = []
        self.response_model = None

    def generate_structured(self, *, messages, response_model, temperature=0.0):
        self.messages = messages
        self.response_model = response_model
        structured = QueryRewriteOutput(
            queries=[
                "EYLF play based learning outcomes",
                "intentional teaching collaborative inquiry",
            ]
        )
        return ModelResponse(
            content=structured.model_dump_json(),
            model="stub-query-rewriter",
            structured=structured,
        )


class StubCrossEncoderReranker:
    def __init__(self) -> None:
        self.calls = []

    def rerank(self, query, chunks):
        self.calls.append({"query": query, "candidate_count": len(chunks)})
        for rank, chunk in enumerate(reversed(chunks), start=1):
            chunk.reranker_score = 1.0 / rank
            chunk.reranker_rank = rank
            chunk.metadata = {**chunk.metadata, "cross_encoder_model": "stub"}
        return list(reversed(chunks))


def test_default_tool_registry_registers_controlled_tools(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    assert [tool.name for tool in registry.list_tools()] == [
        "get_class_context",
        "retrieve_knowledge",
        "query_records",
        "read_draft_artifact",
        "get_daily_context",
        "check_activity_safety",
        "save_observation",
        "save_educational_record",
        "export_records",
        "analyse_learning_records",
    ]


def test_get_class_context_tool_reads_trusted_synthetic_data(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = asyncio.run(
        registry.execute_async(
            "get_class_context",
            {},
            execution_context=ToolExecutionContext(
                teacher_id="teacher-1", class_id="kangaroo-room"
            ),
        )
    )

    assert result.success is True
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.data["name"] == "Kangaroo Room"
    assert result.data["child_count"] == 18
    assert result.trace is not None
    assert result.trace.tool_name == "get_class_context"


def test_get_class_context_rejects_missing_trusted_teacher(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = asyncio.run(
        registry.execute_async(
            "get_class_context",
            {},
            execution_context=ToolExecutionContext(class_id="kangaroo-room"),
        )
    )

    assert result.success is False
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED


def test_retrieve_knowledge_standard_mode_returns_citable_chunks_without_rewrite(tmp_path) -> None:
    retriever = StubPolicyRetriever()
    registry = build_default_tool_registry(
        make_store(tmp_path),
        knowledge_retriever=retriever,
    )

    result = registry.execute(
        "retrieve_knowledge",
        {
            "query": "play based learning",
            "mode": "standard",
            "top_k": 4,
            "knowledge_scope": "eylf",
            "source_type": "official",
        },
    )

    assert result.success is True
    assert result.data["strategy"] == "simple"
    assert result.data["knowledge_scope"] == "eylf"
    assert result.data["search_queries"] == ["play based learning"]
    assert result.data["returned_count"] == 1
    assert result.data["evidence"][0]["evidence_id"] == "E1"
    assert result.data["evidence"][0]["citation"]["source_id"] == "eylf-v2"
    assert retriever.requests[0].query == "play based learning"
    assert retriever.requests[0].top_k == 4
    assert retriever.requests[0].mode is RetrievalMode.HYBRID
    assert retriever.requests[0].filters.source_ids == ["eylf-v2"]
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


def test_check_activity_safety_flags_small_and_scented_sensory_materials(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = registry.execute(
        "check_activity_safety",
        {
                "activity_text": (
                    "Use dried chickpeas, lavender-scented rice, and small animal "
                    "figurines with pinecones in an outdoor storytelling station."
                ),
            "age_group": "3-5",
            "class_size": 18,
        },
    )

    assert result.success is True
    assert result.data["status"] == "needs_revision"
    assert {issue["code"] for issue in result.data["issues"]} >= {
        "small_loose_parts",
        "scented_materials",
        "natural_loose_materials",
        "activity_contains_outdoor",
    }


def test_retrieve_knowledge_deep_mode_rewrites_and_fuses_queries(tmp_path) -> None:
    retriever = StubPolicyRetriever()
    rewriter = StubQueryRewriter()
    reranker = StubCrossEncoderReranker()
    registry = build_default_tool_registry(
        make_store(tmp_path),
        knowledge_retriever=retriever,
        query_rewriter=rewriter,
        knowledge_reranker=reranker,
    )

    result = registry.execute(
        "retrieve_knowledge",
        {
            "query": (
                "Children explore outdoor natural materials through play, "
                "describe textures, and solve problems together."
            ),
            "top_k": 3,
            "mode": "deep",
            "knowledge_scope": "nqs",
        },
    )

    assert result.success is True
    assert result.data["strategy"] == "enhanced"
    assert result.data["reranker"] == "cross_encoder"
    assert result.data["knowledge_scope"] == "nqs"
    assert result.data["evidence"][0]["evidence_id"] == "E1"
    assert len(result.data["search_queries"]) == 3
    assert len(retriever.requests) == 3
    assert all(request.mode is RetrievalMode.HYBRID for request in retriever.requests)
    assert all(
        request.filters.source_ids == ["nqs-guide-qa1"]
        for request in retriever.requests
    )
    assert rewriter.response_model is QueryRewriteOutput
    assert "Question:" in rewriter.messages[1].content
    assert "multi_query_score" in result.data["evidence"][0]["metadata"]
    assert result.data["evidence"][0]["reranker_score"] == 1.0
    assert reranker.calls == [
        {
            "query": (
                "Children explore outdoor natural materials through play, "
                "describe textures, and solve problems together."
            ),
            "candidate_count": 1,
        }
    ]


def test_get_class_context_tool_reports_missing_profile(tmp_path) -> None:
    registry = build_default_tool_registry(make_store(tmp_path))

    result = asyncio.run(
        registry.execute_async(
            "get_class_context",
            {},
            execution_context=ToolExecutionContext(
                teacher_id="teacher-1", class_id="missing-room"
            ),
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.EXECUTION_ERROR
