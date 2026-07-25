"""Schemas for file-based specialist skills."""

from typing import FrozenSet

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.specialist import SpecialistKind


class SkillManifest(BaseModel):
    """Machine-readable metadata stored beside a SKILL.md file."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(min_length=1)
    specialist: SpecialistKind
    description: str = Field(min_length=1)
    required_tool_names: FrozenSet[str] = Field(default_factory=frozenset)
    optional_tool_names: FrozenSet[str] = Field(default_factory=frozenset)
    output_model: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tool_groups(self) -> "SkillManifest":
        overlap = self.required_tool_names & self.optional_tool_names
        if overlap:
            raise ValueError("a skill tool cannot be both required and optional")
        if any(not tool_name.strip() for tool_name in self.tool_names):
            raise ValueError("skill tool names must not be blank")
        return self

    @property
    def tool_names(self) -> FrozenSet[str]:
        return self.required_tool_names | self.optional_tool_names


class LoadedSkill(BaseModel):
    """Validated Skill content returned to an agent."""

    model_config = ConfigDict(frozen=True)

    manifest: SkillManifest
    instructions: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
