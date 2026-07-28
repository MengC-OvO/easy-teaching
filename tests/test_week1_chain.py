from typing import List

from app.schemas import (
    ApprovalStatus,
    CitationMetadata,
    GraphState,
    Intent,
    IntentRouteResult,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    ReActAction,
    ReActDecision,
    ReActState,
    RetrievalMode,
    RetrievalResult,
    RetrievalStats,
    RerankerMode,
    StopReason,
    ToolCall,
    WorkflowStatus,
)
from app.services import EduFlowStore
from app.tools import ToolDefinition, build_default_tool_registry
from app.workflows import build_main_graph, build_planning_workflow


class StubRouter:
    def route(self, user_message: str, *, conversation_context: str = "") -> IntentRouteResult:
        return IntentRouteResult(
            intent=Intent.ACTIVITY_PLANNING,
            confidence=0.95,
            reason="The request asks for an activity plan.",
        )


class SequencedPlanningAgent:
    def __init__(self, decisions: List[ReActDecision]) -> None:
        self.decisions = decisions
        self.calls = 0

    def decide(self, state: ReActState, available_tools: List[ToolDefinition]) -> ReActDecision:
        self.calls += 1
        return self.decisions.pop(0)


class StubPolicyRetriever:
    def retrieve(self, request):
        citation = CitationMetadata(
            source_id="eylf-v2",
            source_type=KnowledgeSourceType.OFFICIAL,
            title="EYLF V2.0",
            version="2.0-2022",
            section="Learning through play",
            page=21,
        )
        return RetrievalResult(
            query=request.query,
            chunks=[
                RetrievedKnowledgeChunk(
                    chunk_id="chunk-001",
                    content="Play-based learning supports children's agency.",
                    citation=citation,
                    content_hash="b" * 64,
                    distance=0.3,
                )
            ],
            stats=RetrievalStats(
                requested_top_k=request.top_k,
                mode=RetrievalMode.BM25,
                reranker=RerankerMode.LEXICAL,
                raw_result_count=1,
                bm25_result_count=1,
                deduplicated_count=1,
                returned_count=1,
                reranked=True,
            ),
        )


def call_tool(name: str, args) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.CALL_TOOL,
        reason=f"Need to call {name}.",
        tool_call=ToolCall(tool_name=name, tool_args=args),
    )


def final_answer(content: str) -> ReActDecision:
    return ReActDecision(
        action=ReActAction.FINAL_ANSWER,
        reason="The draft is ready.",
        final_answer=content,
    )


def make_store(tmp_path) -> EduFlowStore:
    store = EduFlowStore(f"sqlite:///{tmp_path / 'week1-chain.sqlite3'}")
    store.initialize()
    return store


def build_planning_chain(tmp_path, *, approved: bool, agent: SequencedPlanningAgent):
    registry = build_default_tool_registry(
        make_store(tmp_path),
        knowledge_retriever=StubPolicyRetriever(),
    )
    planning_workflow = build_planning_workflow(
        agent=agent,
        registry=registry,
        allowed_tool_names={
            "get_class_profile",
            "retrieve_risk_guidance",
            "save_draft",
        },
        approved=approved,
        required_skill_name=None,
    )
    return build_main_graph(StubRouter(), planning_workflow=planning_workflow)


def planning_decisions(include_final_answer: bool = True) -> List[ReActDecision]:
    decisions = [
        call_tool("get_class_profile", {"class_id": "kangaroo-room"}),
        call_tool("retrieve_risk_guidance", {"query": "program"}),
        call_tool(
            "save_draft",
            {
                "draft_id": "draft-week1-001",
                "idempotency_key": "req-week1:save-draft",
                "draft_type": "activity_plan",
                "title": "Outdoor sensory walk",
                "content": "Synthetic activity plan draft.",
            },
        ),
    ]
    if include_final_answer:
        decisions.append(final_answer("Synthetic activity plan draft."))
    return decisions


def test_week1_chain_routes_to_react_tools_and_waits_for_approval(tmp_path) -> None:
    agent = SequencedPlanningAgent(planning_decisions(include_final_answer=False))
    graph = build_planning_chain(tmp_path, approved=False, agent=agent)

    final_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-week1",
                session_id="session-week1",
                user_message="Plan and save an outdoor sensory activity.",
            )
        )
    )

    assert final_state.workflow_status is WorkflowStatus.WAITING_FOR_APPROVAL
    assert final_state.approval.status is ApprovalStatus.REQUIRED
    planning_trace = final_state.trace[-4]
    assert planning_trace.step == "planning_react"
    assert planning_trace.metadata["stop_reason"] == "approval_required"
    assert [item["tool_name"] for item in planning_trace.metadata["observations"]] == [
        "get_class_profile",
        "retrieve_risk_guidance",
        "save_draft",
    ]
    assert final_state.trace[-3].step == "approval_gate"
    assert final_state.trace[-2].step == "context_update"
    assert final_state.trace[-1].step == "long_memory_update"
    assert agent.calls == 3


def test_week1_chain_routes_to_react_tools_and_saves_draft_when_approved(tmp_path) -> None:
    agent = SequencedPlanningAgent(planning_decisions())
    graph = build_planning_chain(tmp_path, approved=True, agent=agent)

    final_state = GraphState.model_validate(
        graph.invoke(
            GraphState(
                request_id="req-week1-approved",
                session_id="session-week1",
                user_message="Plan and save an outdoor sensory activity.",
            )
        )
    )

    assert final_state.workflow_status is WorkflowStatus.COMPLETED
    assert final_state.draft is not None
    assert final_state.draft.content == "Synthetic activity plan draft."
    planning_trace = final_state.trace[-3]
    assert planning_trace.metadata["stop_reason"] == StopReason.COMPLETED.value
    assert planning_trace.metadata["observations"][-1] == {
        "tool_name": "save_draft",
        "success": True,
        "error_code": None,
    }
    assert final_state.trace[-2].step == "context_update"
    assert final_state.trace[-1].step == "long_memory_update"
    assert agent.calls == 4
