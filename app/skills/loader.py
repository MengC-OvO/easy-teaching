"""Safe reader and validator for registered file-based Skills."""

import hashlib
import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from app.schemas import (
    LoadedSkill,
    SkillManifest,
    SpecialistPermissionDenied,
    SpecialistPermissionPolicy,
)
from app.skills.registry import SkillRegistry


class SkillLoadError(ValueError):
    pass


class SkillLoader:
    """Load SKILL.md and manifest.json from an already trusted registry."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        max_file_bytes: int = 64 * 1024,
    ) -> None:
        self.registry = registry
        self.max_file_bytes = max_file_bytes

    def load(
        self,
        name: str,
        *,
        permission: SpecialistPermissionPolicy,
        registered_tool_names: Iterable[str],
    ) -> LoadedSkill:
        directory = self.registry.resolve(name)
        manifest_text = self._read_file(directory, "manifest.json")
        instructions = self._read_file(directory, "SKILL.md").strip()
        if not instructions:
            raise SkillLoadError(f"Skill instructions are empty: {name}")

        try:
            manifest = SkillManifest.model_validate_json(manifest_text)
        except (ValidationError, ValueError) as error:
            raise SkillLoadError(f"Invalid Skill manifest: {name}") from error

        if manifest.name != name:
            raise SkillLoadError(
                f"Skill manifest name does not match registry name: {manifest.name}"
            )
        permission.require_specialist(manifest.specialist)

        unauthorized_tools = manifest.tool_names - permission.allowed_tool_names
        if unauthorized_tools:
            raise SpecialistPermissionDenied(
                f"{name} requests tools outside {manifest.specialist.value} permissions: "
                + ", ".join(sorted(unauthorized_tools))
            )

        registered_names = frozenset(registered_tool_names)
        unregistered_tools = manifest.tool_names - registered_names
        if unregistered_tools:
            raise SkillLoadError(
                f"{name} requests unregistered tools: "
                + ", ".join(sorted(unregistered_tools))
            )

        return LoadedSkill(
            manifest=manifest,
            instructions=instructions,
            content_hash=hashlib.sha256(instructions.encode("utf-8")).hexdigest(),
        )

    def _read_file(self, directory: Path, filename: str) -> str:
        path = directory / filename
        if path.is_symlink():
            raise SkillLoadError(f"Skill file must not be a symbolic link: {filename}")
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(directory)
        except ValueError as error:
            raise SkillLoadError(f"Skill file escapes its directory: {filename}") from error
        if not resolved_path.is_file():
            raise SkillLoadError(f"Skill file is missing: {filename}")
        if resolved_path.stat().st_size > self.max_file_bytes:
            raise SkillLoadError(f"Skill file exceeds size limit: {filename}")
        try:
            return resolved_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise SkillLoadError(f"Could not read Skill file: {filename}") from error
