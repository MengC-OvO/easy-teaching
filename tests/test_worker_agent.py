import asyncio

from pydantic import BaseModel

from app.agents import BoundedWorkerRunner, WorkerProfile, WorkerRegistry
from app.schemas import (
    ReActAction,
    ReActDecision,
    RiskLevel,
    ToolCall,
    WorkerCall,
    WorkerName,
)
from app.services import ModelResponse
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolRegistry,
    ToolResult,
)


class QueryInput(BaseModel):
    query: str


class QueryOutput(BaseModel):
    answer: str


class SequenceProvider:
    def __init__(self, decisions):
        self.decisions = iter(decisions)

    def generate_structured(self, **kwargs):
        return ModelResponse(content="{}", structured=next(self.decisions))


def _runner(provider, *, allowed_names=frozenset({"internal_search"})):
    tools = ToolRegistry()
    tools.register(
        ToolDefinition(
            name="internal_search",
            description="Search internal evidence.",
            category=ToolCategory.POLICY,
            domain=ToolDomain.INTERNAL,
            parallel_safe=True,
            input_model=QueryInput,
            output_model=QueryOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            handler=lambda args: ToolResult.ok(
                data={"answer": f"Evidence for {args.query}"},
                risk_level=RiskLevel.L0_READ_ONLY,
            ),
        )
    )
    workers = WorkerRegistry(
        [
            WorkerProfile(
                name=WorkerName.INTERNAL_RESEARCH,
                description="Internal only.",
                allowed_tool_names=allowed_names,
                max_steps=3,
            )
        ]
    )
    return BoundedWorkerRunner(
        provider=provider,
        tool_registry=tools,
        worker_registry=workers,
    )


def test_worker_runs_bounded_tool_loop_and_returns_summary() -> None:
    runner = _runner(
        SequenceProvider(
            [
                ReActDecision(
                    action=ReActAction.CALL_TOOL,
                    reason="Need evidence.",
                    tool_call=ToolCall(
                        tool_name="internal_search",
                        tool_args={"query": "play"},
                    ),
                ),
                ReActDecision(
                    action=ReActAction.FINAL_ANSWER,
                    reason="Enough evidence.",
                    final_answer="Play evidence summary.",
                ),
            ]
        )
    )

    observation = asyncio.run(
        runner.run(
            WorkerCall(
                name=WorkerName.INTERNAL_RESEARCH,
                arguments={"task": "Research play."},
                result_key="internal_result",
            ),
            teacher_id="teacher-1",
            class_id="class-1",
            dependency_observations={},
        )
    )

    assert observation.status.value == "completed"
    assert observation.data["summary"] == "Play evidence summary."
    assert observation.data["tool_observations"][0]["success"] is True


def test_worker_permission_error_blocks_tool_outside_allowlist() -> None:
    runner = _runner(
        SequenceProvider(
            [
                ReActDecision(
                    action=ReActAction.CALL_TOOL,
                    reason="Try a blocked tool.",
                    tool_call=ToolCall(
                        tool_name="internal_search",
                        tool_args={"query": "play"},
                    ),
                ),
                ReActDecision(
                    action=ReActAction.FINAL_ANSWER,
                    reason="Report failure.",
                    final_answer="The tool was not available.",
                ),
            ]
        ),
        allowed_names=frozenset({"different_tool"}),
    )

    observation = asyncio.run(
        runner.run(
            WorkerCall(
                name=WorkerName.INTERNAL_RESEARCH,
                arguments={"task": "Research play."},
                result_key="internal_result",
            ),
            teacher_id=None,
            class_id=None,
            dependency_observations={},
        )
    )

    assert observation.status.value == "insufficient"
    assert "没有可用" in observation.data["summary"]


def test_worker_rejects_missing_task_without_calling_model() -> None:
    runner = _runner(SequenceProvider([]))

    observation = asyncio.run(
        runner.run(
            WorkerCall(
                name=WorkerName.INTERNAL_RESEARCH,
                arguments={},
                result_key="internal_result",
            ),
            teacher_id=None,
            class_id=None,
            dependency_observations={},
        )
    )

    assert observation.status.value == "failed"
    assert "task" in observation.error["message"]
