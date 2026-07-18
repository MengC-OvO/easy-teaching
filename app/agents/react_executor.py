from typing import Dict, Iterable, Optional

from app.schemas import Observation, ReActAction, ReActState, StopReason
from app.tools import ToolErrorCode, ToolRegistry, ToolResult


class ReActToolExecutor:
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

    def execute(self, state: ReActState, *, approved: bool = False) -> Dict[str, object]:
        if state.decision is None:
            return {
                "stop_reason": StopReason.MODEL_ERROR,
                "current_step": state.current_step + 1,
            }

        if state.decision.action is ReActAction.FINAL_ANSWER:
            return {
                "final_answer": state.decision.final_answer,
                "stop_reason": StopReason.COMPLETED,
                "current_step": state.current_step + 1,
            }

        tool_call = state.decision.tool_call
        if tool_call is None:
            return {
                "stop_reason": StopReason.MODEL_ERROR,
                "current_step": state.current_step + 1,
            }

        result = self.registry.execute(
            tool_call.tool_name,
            tool_call.tool_args,
            approved=approved,
            allowed_tool_names=self.allowed_tool_names,
        )
        observation = self._result_to_observation(tool_call.tool_name, result)
        next_state = {
            "observations": [observation],
            "current_step": state.current_step + 1,
        }

        if not result.success:
            if (
                result.error
                and result.error.code is ToolErrorCode.PERMISSION_DENIED
                and result.error.recoverable
            ):
                next_state["stop_reason"] = StopReason.APPROVAL_REQUIRED
            elif result.error and not result.error.recoverable:
                next_state["stop_reason"] = StopReason.TOOL_ERROR

        return next_state

    def _result_to_observation(self, tool_name: str, result: ToolResult) -> Observation:
        return Observation(
            tool_name=tool_name,
            success=result.success,
            data=result.data,
            error=result.error.model_dump(mode="json") if result.error else None,
        )
