from app.schemas import (
    Approval,
    ApprovalStatus,
    Citation,
    GraphState,
    RiskLevel,
    TraceEvent,
    WorkflowStatus,
)
from evals.agent_e2e import (
    AgentEvalCheck,
    AgentQualityVerdict,
    _attributed_citation_count,
    _called_capabilities,
    _outcome,
    load_agent_eval_cases,
)


def test_agent_suite_has_broad_but_bounded_coverage() -> None:
    cases = load_agent_eval_cases()

    assert 20 <= len(cases) <= 30
    tags = {tag for case in cases for tag in case.tags}
    assert {
        "activity",
        "rag",
        "observation",
        "approval",
        "records",
        "parallel-tools",
        "parallel-workers",
        "mcp",
        "state-tool-availability",
        "retrieval-budget",
        "multilingual",
        "high-risk",
    } <= tags


def test_capability_extraction_includes_frozen_approval_call() -> None:
    state = GraphState(
        request_id="request-1",
        session_id="session-1",
        user_message="Synthetic request",
        workflow_status=WorkflowStatus.WAITING_FOR_APPROVAL,
        approval=Approval(
            status=ApprovalStatus.REQUIRED,
            risk_level=RiskLevel.L2_CONTROLLED_WRITE,
            action_id="action-1",
            tool_name="save_observation",
            preview={"objective_text": "Synthetic objective observation."},
        ),
        trace=[
            TraceEvent(
                step="merge_observations",
                message="Merged.",
                metadata={
                    "observations": [
                        {"tool_name": "get_class_context", "success": True}
                    ]
                },
            )
        ],
    )

    tools, workers = _called_capabilities(state)

    assert tools == ["get_class_context", "save_observation"]
    assert workers == []
    assert _outcome(state) == "approval"


def test_citation_attribution_requires_the_answer_to_name_the_source() -> None:
    state = GraphState(
        request_id="request-citation",
        session_id="session-citation",
        user_message="Explain agency.",
        citations=[
            Citation(
                source="eylf-v2",
                title="Belonging, Being and Becoming",
                section="Play-based learning and intentionality",
                page=21,
            )
        ],
    )

    assert _attributed_citation_count(
        state,
        "Belonging, Being and Becoming explains that children exercise agency.",
    ) == 1
    assert _attributed_citation_count(state, "Children exercise agency through play.") == 0


def test_quality_verdict_has_a_stable_threshold() -> None:
    passing = AgentQualityVerdict(
        relevance=4,
        completeness=4,
        evidence_support=4,
        safety=5,
        citation_use=4,
        rationale="Grounded and suitable for the task.",
    )
    critical = passing.model_copy(update={"critical_error": True})

    assert passing.normalized_score == 0.84
    assert passing.passed is True
    assert critical.passed is False


def test_telemetry_checks_can_be_non_blocking() -> None:
    check = AgentEvalCheck(
        name="task_type",
        passed=False,
        blocking=False,
        expected="safety_review",
        actual="activity_plan",
    )

    assert check.passed is False
    assert check.blocking is False
