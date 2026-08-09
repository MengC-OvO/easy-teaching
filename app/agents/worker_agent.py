"""受限 Worker Agent：独立上下文、固定工具范围和有界 ReAct 循环。"""

import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Type

from app.schemas import (
    CapabilityObservation,
    CapabilitySource,
    ObservationStatus,
    ReActAction,
    ReActDecision,
    WorkerCall,
    WorkerName,
)
from app.services import ModelMessage, ModelProviderError, ModelResponse, ModelRole
from app.tools import ToolExecutionContext, ToolRegistry


WORKER_SYSTEM_PROMPT = """
You are a bounded research Worker inside EduFlow AU. Work only on the assigned
task and use only the tools listed in the prompt. Choose exactly one action per
turn: call one tool, or return a concise final_answer based on observations.

Never produce the teacher-facing final plan. Never request writes or approvals.
Do not invent missing evidence. If evidence is unavailable, state that clearly.
""".strip()


class WorkerModelProvider(Protocol):
    def generate_structured(
        self,
        *,
        messages: List[ModelMessage],
        response_model: Type[ReActDecision],
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...


@dataclass(frozen=True)
class WorkerProfile:
    name: WorkerName
    description: str
    allowed_tool_names: frozenset[str]
    max_steps: int = 3

    def public_description(self) -> Dict[str, Any]:
        return {
            "name": self.name.value,
            "description": self.description,
            "input": {"task": "具体且不依赖同批其他 Worker 的研究任务"},
            "max_steps": self.max_steps,
        }


DEFAULT_WORKER_PROFILES = (
    WorkerProfile(
        name=WorkerName.INTERNAL_RESEARCH,
        description="深入检索 EYLF、政策与安全证据；不能访问本地儿童数据或网络。",
        allowed_tool_names=frozenset(
            {
                "retrieve_risk_guidance",
                "align_to_eylf_outcomes",
                "check_activity_safety",
            }
        ),
    ),
    WorkerProfile(
        name=WorkerName.LOCAL_CONTEXT,
        description="汇总当前教师有权访问的班级与长期记忆；不能访问公开网络。",
        allowed_tool_names=frozenset(
            {"get_class_profile", "recall_long_term_memory"}
        ),
    ),
    WorkerProfile(
        name=WorkerName.EXTERNAL_RESEARCH,
        description="查询获批的公开信息与天气；不能接收原始儿童或家庭隐私。",
        allowed_tool_names=frozenset(
            {"search_public_resources", "get_public_weather"}
        ),
    ),
)


class WorkerRegistry:
    """只允许执行代码预注册的 Worker Profile。"""

    def __init__(self, profiles: Iterable[WorkerProfile]) -> None:
        self._profiles = {profile.name: profile for profile in profiles}

    def get(self, name: WorkerName) -> Optional[WorkerProfile]:
        return self._profiles.get(name)

    def public_descriptions(self) -> List[Dict[str, Any]]:
        return [profile.public_description() for profile in self._profiles.values()]

    @property
    def names(self) -> frozenset[WorkerName]:
        return frozenset(self._profiles)


class BoundedWorkerRunner:
    def __init__(
        self,
        *,
        provider: WorkerModelProvider,
        tool_registry: ToolRegistry,
        worker_registry: WorkerRegistry,
    ) -> None:
        self.provider = provider
        self.tool_registry = tool_registry
        self.worker_registry = worker_registry

    async def run(
        self,
        call: WorkerCall,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        dependency_observations: Mapping[str, CapabilityObservation],
    ) -> CapabilityObservation:
        profile = self.worker_registry.get(call.name)
        if profile is None:
            return self._failed(call, "Worker 未注册。")

        task = call.arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            return self._failed(call, "Worker 参数必须包含非空 task。")

        available_tools = self.tool_registry.list_tools(
            allowed_tool_names=profile.allowed_tool_names
        )
        if not available_tools:
            return self._insufficient(call, "当前 Worker 没有可用的已注册工具。")

        observations: List[Dict[str, Any]] = []
        for current_step in range(profile.max_steps):
            try:
                decision = await self._decide(
                    task=task,
                    profile=profile,
                    available_tools=available_tools,
                    dependency_observations=dependency_observations,
                    observations=observations,
                    current_step=current_step,
                )
            except (ModelProviderError, TypeError, ValueError) as error:
                return self._failed(call, f"Worker 模型调用失败：{error}")

            if decision.action is ReActAction.FINAL_ANSWER:
                return CapabilityObservation(
                    result_key=call.result_key,
                    capability_name=call.name.value,
                    source_kind=CapabilitySource.WORKER,
                    status=ObservationStatus.COMPLETED,
                    data={
                        "summary": decision.final_answer,
                        "tool_observations": observations,
                    },
                )

            tool_call = decision.tool_call
            assert tool_call is not None
            result = await self.tool_registry.execute_async(
                tool_call.tool_name,
                tool_call.tool_args,
                allowed_tool_names=profile.allowed_tool_names,
                execution_context=ToolExecutionContext(
                    teacher_id=teacher_id,
                    class_id=class_id,
                ),
            )
            observations.append(
                {
                    "tool_name": tool_call.tool_name,
                    "success": result.success,
                    "data": result.data,
                    "error": (
                        result.error.model_dump(mode="json") if result.error else None
                    ),
                }
            )

        return self._insufficient(
            call,
            "Worker 达到最大轮数，未形成最终摘要。",
            observations=observations,
        )

    async def _decide(
        self,
        *,
        task: str,
        profile: WorkerProfile,
        available_tools,
        dependency_observations: Mapping[str, CapabilityObservation],
        observations: List[Dict[str, Any]],
        current_step: int,
    ) -> ReActDecision:
        prompt = {
            "worker": profile.name.value,
            "task": task,
            "step": f"{current_step}/{profile.max_steps}",
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema(),
                }
                for tool in available_tools
            ],
            "dependency_observations": {
                key: value.model_dump(mode="json")
                for key, value in dependency_observations.items()
            },
            "own_tool_observations": observations,
        }
        response = self.provider.generate_structured(
            messages=[
                ModelMessage(role=ModelRole.SYSTEM, content=WORKER_SYSTEM_PROMPT),
                ModelMessage(
                    role=ModelRole.USER,
                    content=json.dumps(prompt, ensure_ascii=False),
                ),
            ],
            response_model=ReActDecision,
            temperature=0.0,
        )
        if inspect.isawaitable(response):
            response = await response
        if not isinstance(response.structured, ReActDecision):
            raise TypeError("Worker provider returned an unexpected result")
        return response.structured

    def _failed(self, call: WorkerCall, message: str) -> CapabilityObservation:
        return CapabilityObservation(
            result_key=call.result_key,
            capability_name=call.name.value,
            source_kind=CapabilitySource.WORKER,
            status=ObservationStatus.FAILED,
            error={"message": message, "recoverable": True},
        )

    def _insufficient(
        self,
        call: WorkerCall,
        message: str,
        *,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> CapabilityObservation:
        return CapabilityObservation(
            result_key=call.result_key,
            capability_name=call.name.value,
            source_kind=CapabilitySource.WORKER,
            status=ObservationStatus.INSUFFICIENT,
            data={"summary": message, "tool_observations": observations or []},
        )
