import json
from typing import List, Optional, Protocol, Type

from app.schemas import Observation, ReActDecision, ReActState
from app.services import (
    ChatCompletionsModelProvider,
    ModelMessage,
    ModelResponse,
    ModelRole,
)
from app.tools import ToolDefinition


REACT_AGENT_SYSTEM_PROMPT = """
You are a constrained ReAct agent inside EduFlow AU.

You must choose exactly one action:
- call_tool: request one available tool with JSON arguments
- final_answer: provide the final answer when enough information is available

Return only valid JSON matching this schema:
{
  "action": "call_tool | final_answer",
  "reason": "short reason",
  "tool_call": {"tool_name": "name", "tool_args": {}} or null,
  "final_answer": "answer text" or null
}

Rules:
- Use only tools listed in the user message.
- Do not invent tool names.
- Call at most one tool per step.
- Prefer read-only context tools before drafting.
- If a tool result says approval is required, stop with a final answer explaining that approval is needed.
- Do not use real child or family private information.
""".strip()


class ReActModelProvider(Protocol):
    def generate_structured(
        self,
        *,
        messages: List[ModelMessage],
        response_model: Type[ReActDecision],
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...


class ReActAgent:
    def __init__(self, provider: Optional[ReActModelProvider] = None) -> None:
        self.provider = provider or ChatCompletionsModelProvider()

    def decide(self, state: ReActState, available_tools: List[ToolDefinition]) -> ReActDecision:
        response = self.provider.generate_structured(
            messages=[
                ModelMessage(role=ModelRole.SYSTEM, content=REACT_AGENT_SYSTEM_PROMPT),
                ModelMessage(
                    role=ModelRole.USER,
                    content=self._build_user_prompt(state, available_tools),
                ),
            ],
            response_model=ReActDecision,
            temperature=0.0,
        )

        if not isinstance(response.structured, ReActDecision):
            raise TypeError("ReAct agent provider returned an unexpected structured result")
        return response.structured

    def _build_user_prompt(
        self,
        state: ReActState,
        available_tools: List[ToolDefinition],
    ) -> str:
        return "\n\n".join(
            [
                f"Teacher request:\n{state.user_message}",
                (
                    "Relevant conversation context:\n"
                    f"{state.conversation_context or '[No prior conversation context.]'}"
                ),
                f"Step budget:\ncurrent_step={state.current_step}, max_steps={state.max_steps}",
                self._format_skill(state),
                "Available tools:\n" + self._format_tools(available_tools),
                "Previous observations:\n" + self._format_observations(state.observations),
            ]
        )

    def _format_skill(self, state: ReActState) -> str:
        if state.required_skill_name is None:
            return "Required Skill:\n[No required Skill.]"
        if state.loaded_skill is None:
            return (
                "Required Skill:\n"
                f"Load {state.required_skill_name!r} with load_skill before doing "
                "anything else. Do not provide a final answer before it is loaded."
            )
        return "\n".join(
            [
                "Loaded Skill instructions:",
                state.loaded_skill.instructions,
                "Loaded Skill manifest:",
                state.loaded_skill.manifest.model_dump_json(),
                "Final output contract:",
                (
                    "Set final_answer to a JSON string matching this JSON Schema. "
                    "Do not wrap the JSON in Markdown fences."
                ),
                json.dumps(state.final_output_schema, ensure_ascii=False),
            ]
        )

    def _format_tools(self, tools: List[ToolDefinition]) -> str:
        tool_summaries = []
        for tool in tools:
            tool_summaries.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "risk_level": tool.risk_level.value,
                    "permission": tool.permission.value,
                    "input_schema": tool.input_schema(),
                }
            )
        return json.dumps(tool_summaries, ensure_ascii=False)

    def _format_observations(self, observations: List[Observation]) -> str:
        if not observations:
            return "[]"
        return json.dumps(
            [observation.model_dump() for observation in observations],
            ensure_ascii=False,
        )
