from typing import Dict, Iterable, Optional, Set

from app.schemas import LoadedSkill, Observation, ReActAction, ReActState, RiskLevel, StopReason
from app.tools import ToolErrorCode, ToolExecutionContext, ToolRegistry, ToolResult


class ReActToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        allowed_tool_names: Optional[Iterable[str]] = None,
        required_skill_name: Optional[str] = None,
    ) -> None:
        self.registry = registry
        self.allowed_tool_names = (
            set(allowed_tool_names) if allowed_tool_names is not None else None
        )
        self.required_skill_name = required_skill_name

    def execute(self, state: ReActState, *, approved: bool = False) -> Dict[str, object]:
        if state.decision is None:
            return {
                "stop_reason": StopReason.MODEL_ERROR,
                "current_step": state.current_step + 1,
            }

        if state.decision.action is ReActAction.FINAL_ANSWER:
            skill_stop_reason = self._validate_final_skill_state(state)
            if skill_stop_reason is StopReason.SKILL_REQUIREMENTS_MISSING:
                return self._missing_skill_requirements_update(state)
            if skill_stop_reason is not None:
                return {
                    "stop_reason": skill_stop_reason,
                    "current_step": state.current_step + 1,
                }
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

        if (
            self.required_skill_name is not None
            and state.loaded_skill is None
            and tool_call.tool_name == "load_skill"
            and tool_call.tool_args.get("skill_name") != self.required_skill_name
        ):
            result = ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=(
                    "This workflow requires Skill: "
                    f"{self.required_skill_name}"
                ),
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
                details={"required_skill_name": self.required_skill_name},
            )
            return self._tool_result_update(state, tool_call.tool_name, result)

        result = self.registry.execute(
            tool_call.tool_name,
            tool_call.tool_args,
            approved=approved,
            allowed_tool_names=self.effective_allowed_tool_names(state),
            execution_context=ToolExecutionContext(
                teacher_id=state.teacher_id,
                class_id=state.class_id,
            ),
        )
        next_state = self._tool_result_update(state, tool_call.tool_name, result)
        if (
            result.success
            and tool_call.tool_name == "load_skill"
            and self.required_skill_name is not None
        ):
            loaded_skill = LoadedSkill.model_validate(result.data)
            next_state["loaded_skill"] = loaded_skill
            next_state["observations"] = [
                Observation(
                    tool_name="load_skill",
                    success=True,
                    data={
                        "skill_name": loaded_skill.manifest.name,
                        "version": loaded_skill.manifest.version,
                        "content_hash": loaded_skill.content_hash,
                    },
                )
            ]
        return next_state

    def effective_allowed_tool_names(self, state: ReActState) -> Optional[Set[str]]:
        if self.required_skill_name is None:
            return (
                set(self.allowed_tool_names)
                if self.allowed_tool_names is not None
                else None
            )
        base_allowed = (
            set(self.allowed_tool_names)
            if self.allowed_tool_names is not None
            else {tool.name for tool in self.registry.list_tools()}
        )
        if state.loaded_skill is None:
            return base_allowed & {"load_skill"}
        if state.loaded_skill.manifest.name != self.required_skill_name:
            return set()
        return base_allowed - {"load_skill"}

    def _validate_final_skill_state(
        self,
        state: ReActState,
    ) -> Optional[StopReason]:
        if self.required_skill_name is None:
            return None
        if (
            state.loaded_skill is None
            or state.loaded_skill.manifest.name != self.required_skill_name
        ):
            return StopReason.SKILL_REQUIRED
        successful_tools = {
            observation.tool_name
            for observation in state.observations
            if observation.success
        }
        missing_tools = (
            state.loaded_skill.manifest.required_tool_names - successful_tools
        )
        if missing_tools:
            return StopReason.SKILL_REQUIREMENTS_MISSING
        return None

    def _missing_skill_requirements_update(
        self,
        state: ReActState,
    ) -> Dict[str, object]:
        """Return recoverable feedback when the model answers too early.

        Required Skill tools are a code-enforced workflow contract.  A model
        may occasionally attempt a final answer before satisfying that
        contract; keep the ReAct loop alive so it can call the missing tools on
        the next step instead of failing the whole teacher request.
        """
        successful_tools = {
            observation.tool_name
            for observation in state.observations
            if observation.success
        }
        missing_tools = sorted(
            state.loaded_skill.manifest.required_tool_names - successful_tools
        )
        return {
            "observations": [
                Observation(
                    tool_name="skill_requirements_check",
                    success=False,
                    error={
                        "code": StopReason.SKILL_REQUIREMENTS_MISSING.value,
                        "message": (
                            "Final answer rejected. Call every missing required "
                            "tool before answering."
                        ),
                        "recoverable": True,
                        "details": {"missing_tool_names": missing_tools},
                    },
                )
            ],
            "current_step": state.current_step + 1,
        }

    def _tool_result_update(
        self,
        state: ReActState,
        tool_name: str,
        result: ToolResult,
    ) -> Dict[str, object]:
        observation = self._result_to_observation(tool_name, result)
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
