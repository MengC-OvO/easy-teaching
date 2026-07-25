"""Read-only tool for loading a registered specialist Skill."""

from typing import Iterable, Optional

from pydantic import BaseModel, Field

from app.schemas import (
    LoadedSkill,
    RiskLevel,
    SpecialistKind,
    SpecialistPermissionDenied,
    SpecialistPermissionPolicy,
    get_specialist_permission,
)
from app.skills import (
    SkillLoadError,
    SkillLoader,
    SkillNotFoundError,
    build_default_skill_registry,
)
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolErrorCode,
    ToolPermission,
    ToolResult,
)


class LoadSkillInput(BaseModel):
    skill_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")


def build_load_skill_tool(
    *,
    loader: Optional[SkillLoader] = None,
    permission: Optional[SpecialistPermissionPolicy] = None,
    registered_tool_names: Iterable[str],
) -> ToolDefinition:
    resolved_loader = loader or SkillLoader(build_default_skill_registry())
    resolved_permission = permission or get_specialist_permission(
        SpecialistKind.PLANNING
    )
    resolved_permission.require_specialist(SpecialistKind.PLANNING)
    registered_names = frozenset(registered_tool_names)

    def handler(input_data: BaseModel) -> ToolResult:
        data = LoadSkillInput.model_validate(input_data)
        try:
            loaded_skill = resolved_loader.load(
                data.skill_name,
                permission=resolved_permission,
                registered_tool_names=registered_names,
            )
        except SpecialistPermissionDenied as error:
            return ToolResult.fail(
                code=ToolErrorCode.PERMISSION_DENIED,
                message=str(error),
                risk_level=RiskLevel.L3_FORBIDDEN,
                recoverable=False,
                details={"skill_name": data.skill_name},
            )
        except (SkillNotFoundError, SkillLoadError) as error:
            return ToolResult.fail(
                code=ToolErrorCode.VALIDATION_ERROR,
                message=str(error),
                risk_level=RiskLevel.L0_READ_ONLY,
                recoverable=True,
                details={"skill_name": data.skill_name},
            )

        return ToolResult.ok(
            data=loaded_skill.model_dump(mode="json"),
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="load_skill",
        description=(
            "Load the instructions and manifest for a registered specialist Skill "
            "by its safe registry name."
        ),
        category=ToolCategory.SYSTEM,
        input_model=LoadSkillInput,
        output_model=LoadedSkill,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )
