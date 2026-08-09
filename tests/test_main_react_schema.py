import pytest
from pydantic import ValidationError

from app.schemas import (
    CapabilityCall,
    CapabilityObservation,
    CapabilitySource,
    MainDecision,
    ObservationStatus,
    WorkerCall,
    WorkerName,
)


def test_main_decision_accepts_one_or_many_tool_calls() -> None:
    decision = MainDecision(
        reason="两个查询彼此独立。",
        tool_calls=[
            CapabilityCall(
                name="get_weather",
                arguments={"location": "Sydney"},
                result_key="weather",
            ),
            CapabilityCall(
                name="search_eylf",
                arguments={"query": "Outcome 4"},
                result_key="eylf",
            ),
        ],
    )

    assert len(decision.current_calls) == 2
    assert decision.current_calls[0].name == "get_weather"


def test_main_decision_accepts_parallel_worker_calls() -> None:
    decision = MainDecision(
        reason="三项深度研究互不依赖。",
        worker_calls=[
            WorkerCall(
                name=WorkerName.INTERNAL_RESEARCH,
                result_key="internal",
            ),
            WorkerCall(
                name=WorkerName.LOCAL_CONTEXT,
                result_key="local",
            ),
        ],
    )

    assert len(decision.worker_calls) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"reason": "没有选择。"},
        {
            "reason": "混合了两种选择。",
            "tool_calls": [
                {"name": "search_eylf", "result_key": "eylf"}
            ],
            "final_answer": "完成。",
        },
        {
            "reason": "结果键冲突。",
            "tool_calls": [
                {"name": "first", "result_key": "same"},
                {"name": "second", "result_key": "same"},
            ],
        },
    ],
)
def test_main_decision_rejects_invalid_combinations(payload) -> None:
    with pytest.raises(ValidationError):
        MainDecision.model_validate(payload)


def test_call_deduplicates_needs_and_has_stable_signature() -> None:
    call = CapabilityCall(
        name="search_eylf",
        arguments={"b": 2, "a": 1},
        needs=["context", "context"],
        result_key="eylf",
    )

    assert call.needs == ["context"]
    assert call.signature() == '{"arguments": {"a": 1, "b": 2}, "name": "search_eylf"}'


def test_observation_marks_completed_and_insufficient_as_available() -> None:
    completed = CapabilityObservation(
        result_key="weather",
        capability_name="get_weather",
        source_kind=CapabilitySource.MCP,
        status=ObservationStatus.COMPLETED,
    )
    failed = completed.model_copy(update={"status": ObservationStatus.FAILED})

    assert completed.is_available is True
    assert failed.is_available is False

