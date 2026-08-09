"""Main ReAct 决策验证和受控工具执行。"""

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional

from app.schemas import (
    CapabilityCall,
    CapabilityObservation,
    CapabilitySource,
    MainDecision,
    ObservationStatus,
    WorkerName,
)
from app.tools import (
    ToolDomain,
    ToolExecutionContext,
    ToolKind,
    ToolPermission,
    ToolRegistry,
    ToolResult,
)


class ExecutionRoute(str, Enum):
    SINGLE_TOOL = "single_tool"
    PARALLEL_TOOLS = "parallel_tools"
    PARALLEL_WORKERS = "parallel_workers"
    FEEDBACK = "decision_feedback"
    CLARIFICATION = "clarification"
    FINAL = "finalize_draft"


@dataclass(frozen=True)
class DecisionValidation:
    route: ExecutionRoute
    feedback: Optional[CapabilityObservation] = None


class MainDecisionValidator:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_tool_names: Optional[Iterable[str]] = None,
        allowed_worker_names: Optional[Iterable[WorkerName]] = None,
        max_repeated_calls: int = 2,
    ) -> None:
        self.registry = registry
        self.allowed_tool_names = (
            set(allowed_tool_names) if allowed_tool_names is not None else None
        )
        self.max_repeated_calls = max_repeated_calls
        self.allowed_worker_names = (
            set(allowed_worker_names) if allowed_worker_names is not None else None
        )

    def validate(
        self,
        decision: MainDecision,
        *,
        observations: Dict[str, CapabilityObservation],
        repeated_call_counts: Dict[str, int],
    ) -> DecisionValidation:
        if decision.final_answer:
            return DecisionValidation(ExecutionRoute.FINAL)
        if decision.clarification_question:
            return DecisionValidation(ExecutionRoute.CLARIFICATION)
        if decision.worker_calls:
            if len(decision.worker_calls) < 2:
                return self._feedback(
                    "单一深度任务应由 Main ReAct 继续调用普通工具完成。"
                )
            for call in decision.worker_calls:
                if (
                    self.allowed_worker_names is not None
                    and call.name not in self.allowed_worker_names
                ):
                    return self._feedback(f"Worker 未注册：{call.name.value}")
                if (
                    call.name is WorkerName.EXTERNAL_RESEARCH
                    and self._contains_private_keys(call.arguments)
                ):
                    return self._feedback("外部 Worker 参数可能包含隐私信息。")
                if call.name is WorkerName.EXTERNAL_RESEARCH:
                    for dependency in call.needs:
                        source = observations.get(dependency)
                        tool = (
                            self.registry.get(source.capability_name)
                            if source is not None
                            else None
                        )
                        if tool is not None and tool.domain is ToolDomain.LOCAL:
                            return self._feedback(
                                "外部 Worker 不能接收本地数据 Observation。"
                            )
            missing = self._missing_dependencies(
                decision.worker_calls,
                observations,
            )
            if missing:
                return self._feedback(
                    "Worker 依赖尚未满足：" + ", ".join(sorted(missing))
                )
            return DecisionValidation(ExecutionRoute.PARALLEL_WORKERS)

        calls = decision.tool_calls
        missing = self._missing_dependencies(calls, observations)
        if missing:
            return self._feedback(
                "工具依赖尚未满足：" + ", ".join(sorted(missing))
            )

        for call in calls:
            tool = self.registry.get(call.name)
            if tool is None:
                return self._feedback(f"能力未注册：{call.name}")
            if (
                self.allowed_tool_names is not None
                and call.name not in self.allowed_tool_names
            ):
                return self._feedback(f"当前主流程不允许调用：{call.name}")
            if tool.permission is not ToolPermission.AUTO_EXECUTE:
                return self._feedback(f"草稿流程禁止受控或禁止工具：{call.name}")
            if repeated_call_counts.get(call.signature(), 0) >= self.max_repeated_calls:
                return self._feedback(f"相同调用已经达到重复上限：{call.name}")
            if tool.domain is ToolDomain.EXTERNAL and self._contains_private_keys(
                call.arguments
            ):
                return self._feedback(f"外部能力参数可能包含隐私信息：{call.name}")

        if len(calls) == 1:
            return DecisionValidation(ExecutionRoute.SINGLE_TOOL)
        if any(not self.registry.get(call.name).parallel_safe for call in calls):
            return self._feedback("当前批次包含不可并发工具，请顺序调用。")
        return DecisionValidation(ExecutionRoute.PARALLEL_TOOLS)

    def _missing_dependencies(self, calls, observations) -> set:
        available = {
            key
            for key, observation in observations.items()
            if observation.is_available
        }
        return {
            needed
            for call in calls
            for needed in call.needs
            if needed not in available
        }

    def _contains_private_keys(self, value) -> bool:
        private_markers = {
            "child_name",
            "family_name",
            "raw_observation",
            "email",
            "phone",
        }
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in private_markers:
                    return True
                if self._contains_private_keys(nested):
                    return True
        if isinstance(value, list):
            return any(self._contains_private_keys(item) for item in value)
        if isinstance(value, str):
            if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value):
                return True
            if re.search(r"(?:\+?61|0)[ -]?[2-478](?:[ -]?\d){8}", value):
                return True
        return False

    def _feedback(self, message: str) -> DecisionValidation:
        return self.feedback(message)

    def feedback(self, message: str) -> DecisionValidation:
        return DecisionValidation(
            route=ExecutionRoute.FEEDBACK,
            feedback=CapabilityObservation(
                result_key="decision_feedback",
                capability_name="decision_validator",
                source_kind=CapabilitySource.SYSTEM,
                status=ObservationStatus.REJECTED,
                error={"message": message, "recoverable": True},
            ),
        )


class MainToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_tool_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.registry = registry
        self.allowed_tool_names = (
            set(allowed_tool_names) if allowed_tool_names is not None else None
        )

    async def execute_one(
        self,
        call: CapabilityCall,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> CapabilityObservation:
        result = await self.registry.execute_async(
            call.name,
            call.arguments,
            allowed_tool_names=self.allowed_tool_names,
            execution_context=ToolExecutionContext(
                teacher_id=teacher_id,
                class_id=class_id,
            ),
        )
        return self._to_observation(call, result)

    async def execute_many(
        self,
        calls: List[CapabilityCall],
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> List[CapabilityObservation]:
        return list(
            await asyncio.gather(
                *[
                    self.execute_one(
                        call,
                        teacher_id=teacher_id,
                        class_id=class_id,
                    )
                    for call in calls
                ]
            )
        )

    def _to_observation(
        self,
        call: CapabilityCall,
        result: ToolResult,
    ) -> CapabilityObservation:
        tool = self.registry.get(call.name)
        source = (
            CapabilitySource.MCP
            if tool is not None and tool.kind is ToolKind.MCP
            else CapabilitySource.TOOL
        )
        return CapabilityObservation(
            result_key=call.result_key,
            capability_name=call.name,
            source_kind=source,
            status=(
                ObservationStatus.COMPLETED
                if result.success
                else ObservationStatus.FAILED
            ),
            data=result.data,
            error=(result.error.model_dump(mode="json") if result.error else None),
        )
