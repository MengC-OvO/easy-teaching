import json

from evals.final_agent_suite import (
    DEFAULT_FINAL_CASES_PATH,
    _parse_sse,
    _trace,
    load_cases,
)


def test_final_suite_is_independent_and_bounded() -> None:
    cases = load_cases()

    assert len(cases) == 100
    assert sum(len(case.turns) for case in cases) == 118
    assert sum(
        turn.expected.judge_quality for case in cases for turn in case.turns
    ) == 58
    assert {case.category for case in cases} == {
        "activity_safety",
        "rag_grounding",
        "records",
        "controlled_writes",
        "communication",
        "orchestration",
        "security",
        "multi_turn",
    }
    assert "agent_e2e_cases" not in DEFAULT_FINAL_CASES_PATH.name


def test_final_suite_case_ids_are_unique_and_placeholders_are_bounded() -> None:
    payload = json.loads(DEFAULT_FINAL_CASES_PATH.read_text(encoding="utf-8"))
    ids = [item["id"] for item in payload]

    assert len(ids) == len(set(ids))
    assert all(len(turn["message"]) <= 20_000 for item in payload for turn in item["turns"])


def test_final_suite_extracts_tools_workers_and_context_metrics() -> None:
    body = "\n".join(
        [
            "data: "
            + json.dumps(
                {
                    "data": {
                        "origin": "graph",
                        "step": "main_react",
                        "metadata": {
                            "tool_schema_chars": 1200,
                            "observation_view_chars": 300,
                            "conversation_context_chars": 450,
                        },
                    }
                }
            ),
            "data: "
            + json.dumps(
                {
                    "data": {
                        "origin": "graph",
                        "step": "merge_observations",
                        "metadata": {
                            "observations": [
                                {"tool_name": "query_records"},
                                {"tool_name": "curriculum_research_worker"},
                            ]
                        },
                    }
                }
            ),
        ]
    )

    metrics = _trace(_parse_sse(body))

    assert metrics["tools"] == ["query_records"]
    assert metrics["workers"] == ["curriculum_research_worker"]
    assert metrics["tool_schema_chars"] == [1200]
    assert metrics["observation_view_chars"] == [300]
    assert metrics["context_chars"] == [450]
