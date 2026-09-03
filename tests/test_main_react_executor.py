import asyncio
import time

from pydantic import BaseModel

from app.agents import ExecutionRoute, MainDecisionValidator, MainToolExecutor
from app.schemas import (
    CapabilityCall,
    CapabilityObservation,
    CapabilitySource,
    MainDecision,
    ObservationStatus,
    RiskLevel,
    WorkerCall,
    WorkerName,
)
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolRegistry,
    ToolResult,
)


class EchoInput(BaseModel):
    text: str


class EchoOutput(BaseModel):
    text: str


def make_registry(*, delay: float = 0.0) -> ToolRegistry:
    def handler(input_data: BaseModel) -> ToolResult:
        data = EchoInput.model_validate(input_data)
        if delay:
            time.sleep(delay)
        return ToolResult.ok(
            data={"text": data.text},
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo text.",
            category=ToolCategory.SYSTEM,
            input_model=EchoInput,
            output_model=EchoOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            domain=ToolDomain.INTERNAL,
            parallel_safe=True,
            handler=handler,
        )
    )
    return registry


def test_validator_routes_single_and_parallel_tools() -> None:
    validator = MainDecisionValidator(make_registry())
    single = MainDecision(
        reason="一次调用。",
        tool_calls=[CapabilityCall(name="echo", result_key="one")],
    )
    parallel = MainDecision(
        reason="两个独立调用。",
        tool_calls=[
            CapabilityCall(name="echo", arguments={"text": "a"}, result_key="a"),
            CapabilityCall(name="echo", arguments={"text": "b"}, result_key="b"),
        ],
    )

    assert validator.validate(single, observations={}, repeated_call_counts={}).route is ExecutionRoute.SINGLE_TOOL
    assert validator.validate(parallel, observations={}, repeated_call_counts={}).route is ExecutionRoute.PARALLEL_TOOLS


def test_validator_rejects_single_worker() -> None:
    validator = MainDecisionValidator(make_registry())
    one_worker = MainDecision(
        reason="只有一个Worker。",
        worker_calls=[
            WorkerCall(
                name=WorkerName.CURRICULUM_RESEARCH,
                arguments={"task": "research", "research_questions": ["a", "b"]},
                result_key="research",
            )
        ],
    )

    assert validator.validate(one_worker, observations={}, repeated_call_counts={}).route is ExecutionRoute.FEEDBACK


def test_validator_accepts_two_independent_deep_workers() -> None:
    validator = MainDecisionValidator(make_registry())
    observations = {
        "context": CapabilityObservation(
            result_key="context",
            capability_name="echo",
            source_kind=CapabilitySource.TOOL,
            status=ObservationStatus.COMPLETED,
        )
    }
    decision = MainDecision(
        reason="两个深度任务独立。",
        worker_calls=[
            WorkerCall(
                name=WorkerName.CURRICULUM_RESEARCH,
                arguments={"task": "curriculum", "research_questions": ["a", "b"]},
                result_key="internal",
            ),
            WorkerCall(
                name=WorkerName.RECORD_CONTEXT,
                arguments={"task": "records", "research_questions": ["c", "d"]},
                result_key="external",
            ),
        ],
    )

    result = validator.validate(
        decision,
        observations=observations,
        repeated_call_counts={},
    )

    assert result.route is ExecutionRoute.PARALLEL_WORKERS


def test_tool_executor_runs_one_and_many() -> None:
    executor = MainToolExecutor(make_registry())
    one = asyncio.run(
        executor.execute_one(
            CapabilityCall(
                name="echo",
                arguments={"text": "one"},
                result_key="one",
            ),
            teacher_id="teacher-1",
            class_id=None,
        )
    )
    many = asyncio.run(
        executor.execute_many(
            [
                CapabilityCall(
                    name="echo",
                    arguments={"text": "a"},
                    result_key="a",
                ),
                CapabilityCall(
                    name="echo",
                    arguments={"text": "b"},
                    result_key="b",
                ),
            ],
            teacher_id="teacher-1",
            class_id=None,
        )
    )

    assert one.data == {"text": "one"}
    assert [item.result_key for item in many] == ["a", "b"]


def test_validator_resolves_permission_from_current_tool_arguments() -> None:
    registry = make_registry()
    registry.register(
        ToolDefinition(
            name="gateway",
            description="Discover or execute a dynamic tool.",
            category=ToolCategory.SYSTEM,
            input_model=EchoInput,
            output_model=EchoOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            permission_resolver=lambda arguments: (
                ToolPermission.REQUIRE_APPROVAL
                if arguments.get("text") == "write"
                else ToolPermission.AUTO_EXECUTE
            ),
            risk_resolver=lambda arguments: (
                RiskLevel.L2_CONTROLLED_WRITE
                if arguments.get("text") == "write"
                else RiskLevel.L0_READ_ONLY
            ),
            handler=lambda data: ToolResult.ok(
                data={"text": data.text},
                risk_level=RiskLevel.L0_READ_ONLY,
            ),
        )
    )
    validator = MainDecisionValidator(registry)

    discover = MainDecision(
        reason="discover",
        tool_calls=[
            CapabilityCall(
                name="gateway", arguments={"text": "discover"}, result_key="d"
            )
        ],
    )
    write = MainDecision(
        reason="write",
        tool_calls=[
            CapabilityCall(
                name="gateway", arguments={"text": "write"}, result_key="w"
            )
        ],
    )

    assert validator.validate(
        discover, observations={}, repeated_call_counts={}
    ).route is ExecutionRoute.SINGLE_TOOL
    assert validator.validate(
        write, observations={}, repeated_call_counts={}
    ).route is ExecutionRoute.APPROVAL

