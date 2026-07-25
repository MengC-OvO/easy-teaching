"""Name-based registry for trusted local Skill directories."""

from pathlib import Path
from typing import Dict, Optional


DEFAULT_SKILL_ROOT = Path(__file__).resolve().parent


class SkillRegistryError(ValueError):
    pass


class DuplicateSkillError(SkillRegistryError):
    pass


class SkillNotFoundError(SkillRegistryError):
    pass


class SkillRegistry:
    """Resolve registered Skill names without accepting arbitrary file paths."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or DEFAULT_SKILL_ROOT).resolve()
        self._skills: Dict[str, Path] = {}

    def register(self, name: str, relative_directory: str) -> None:
        if name in self._skills:
            raise DuplicateSkillError(f"Skill already registered: {name}")
        if not name or any(character in name for character in ("/", "\\", ".")):
            raise SkillRegistryError(f"Invalid Skill name: {name}")

        directory = (self.root / relative_directory).resolve()
        try:
            directory.relative_to(self.root)
        except ValueError as error:
            raise SkillRegistryError(
                f"Skill directory escapes the registry root: {relative_directory}"
            ) from error
        if not directory.is_dir():
            raise SkillRegistryError(f"Skill directory does not exist: {relative_directory}")
        self._skills[name] = directory

    def resolve(self, name: str) -> Path:
        try:
            return self._skills[name]
        except KeyError as error:
            raise SkillNotFoundError(f"Skill is not registered: {name}") from error

    def list_names(self):
        return sorted(self._skills)


def build_default_skill_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register("activity_planning", "activity_planning")
    return registry
