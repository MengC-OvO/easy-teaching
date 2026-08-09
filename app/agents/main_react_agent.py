"""统一 Main ReAct 决策器。"""

import json
import inspect
from typing import Any, Dict, List, Protocol, Type

from app.schemas import CapabilityObservation, MainDecision
from app.services import ModelMessage, ModelResponse, ModelRole
from app.tools import ToolDefinition


MAIN_REACT_SYSTEM_PROMPT = """
You are the Main ReAct agent for EduFlow AU.

Decide only the current executable action or current executable batch. Never
produce a complete future plan.

Return JSON matching MainDecision and choose exactly one:
- tool_calls: one Tool/MCP call, or several simple independent calls;
- worker_calls: at least two independent deep research tasks;
- final_answer: the teacher-facing draft when evidence is sufficient;
- clarification_question: one question when required information is missing.

Rules:
- Use only registered names shown in the prompt.
- A single deep task stays in Main: call its ordinary tools over multiple ReAct
  turns instead of delegating one Worker.
- Use Worker calls only when at least two deep tasks are mutually independent.
- Do not mix tool_calls and worker_calls in one decision.
- Every call must list the observation keys it needs.
- Every needed key must already exist before this decision starts.
- Calls in one batch must not depend on one another's future results.
- If independence is uncertain, return only one safe call.
- Never request writes, approvals, sending, diagnosis, medical advice, legal
  conclusions, or raw private child/family information.
- Failed or insufficient observations are limitations, not facts to invent.
- The final answer must be clearly presented as a draft where applicable.
""".strip()


class MainReActProvider(Protocol):
    def generate_structured(
        self,
        *,
        messages: List[ModelMessage],
        response_model: Type[MainDecision],
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...


class MainReActAgent:
    def __init__(self, provider: MainReActProvider) -> None:
        self.provider = provider

    async def decide(
        self,
        *,
        user_message: str,
        conversation_context: str,
        observations: Dict[str, CapabilityObservation],
        available_tools: List[ToolDefinition],
        available_workers: List[Dict[str, Any]],
        current_step: int,
        max_steps: int,
    ) -> MainDecision:
        """在线程中调用当前同步 Provider，向主图提供异步接口。"""

        response = self.provider.generate_structured(
            messages=[
                ModelMessage(role=ModelRole.SYSTEM, content=MAIN_REACT_SYSTEM_PROMPT),
                ModelMessage(
                    role=ModelRole.USER,
                    content=self._build_user_prompt(
                        user_message=user_message,
                        conversation_context=conversation_context,
                        observations=observations,
                        available_tools=available_tools,
                        available_workers=available_workers,
                        current_step=current_step,
                        max_steps=max_steps,
                    ),
                ),
            ],
            response_model=MainDecision,
            temperature=0.0,
        )
        if inspect.isawaitable(response):
            response = await response
        if not isinstance(response.structured, MainDecision):
            raise TypeError("Main ReAct provider returned an unexpected result")
        return response.structured

    def _build_user_prompt(
        self,
        *,
        user_message: str,
        conversation_context: str,
        observations: Dict[str, CapabilityObservation],
        available_tools: List[ToolDefinition],
        available_workers: List[Dict[str, Any]],
        current_step: int,
        max_steps: int,
    ) -> str:
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "domain": tool.domain.value,
                "parallel_safe": tool.parallel_safe,
                "input_schema": tool.input_schema(),
            }
            for tool in available_tools
        ]
        observation_payload = {
            key: value.model_dump(mode="json")
            for key, value in observations.items()
        }
        return "\n\n".join(
            [
                f"Teacher request:\n{user_message}",
                (
                    "Relevant conversation context:\n"
                    f"{conversation_context or '[No prior context.]'}"
                ),
                f"Step budget:\n{current_step}/{max_steps}",
                "Available Tools/MCP:\n"
                + json.dumps(tools, ensure_ascii=False),
                "Available Worker profiles:\n"
                + json.dumps(available_workers, ensure_ascii=False),
                "Current observations:\n"
                + json.dumps(observation_payload, ensure_ascii=False),
            ]
        )
