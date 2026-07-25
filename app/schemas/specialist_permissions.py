"""Permission policies for isolated specialist workflows."""

from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.specialist import SpecialistKind


class ForbiddenSpecialistAction(str, Enum):
    """Actions that a specialist must never perform autonomously."""

    REAL_WORLD_SEND = "real_world_send"
    RAW_PII_OUTPUT = "raw_pii_output"
    CHILD_DIAGNOSIS = "child_diagnosis"
    MEDICAL_ADVICE = "medical_advice"
    LEGAL_COMPLIANCE_CONCLUSION = "legal_compliance_conclusion"
    UNAPPROVED_WRITE = "unapproved_write"


class SpecialistPermissionDenied(PermissionError):
    """Raised when code requests a capability outside a specialist boundary."""


class SpecialistPermissionPolicy(BaseModel):
    """Immutable capability boundary for one specialist workflow."""

    model_config = ConfigDict(frozen=True)

    specialist: SpecialistKind
    allowed_tool_names: FrozenSet[str] = Field(default_factory=frozenset)
    max_steps: int = Field(ge=1)
    forbidden_actions: FrozenSet[ForbiddenSpecialistAction] = Field(
        default_factory=frozenset
    )

    @field_validator("allowed_tool_names")
    @classmethod
    def validate_tool_names(cls, names: FrozenSet[str]) -> FrozenSet[str]:
        if any(not name.strip() for name in names):
            raise ValueError("allowed tool names must not be blank")
        return names

    def require_specialist(self, specialist: SpecialistKind) -> None:
        if self.specialist is not specialist:
            raise SpecialistPermissionDenied(
                f"{specialist.value} workflow cannot use "
                f"{self.specialist.value} permissions"
            )

    def require_tool(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tool_names:
            raise SpecialistPermissionDenied(
                f"{self.specialist.value} specialist cannot use tool: {tool_name}"
            )

    def require_step(self, step: int) -> None:
        if step < 0 or step >= self.max_steps:
            raise SpecialistPermissionDenied(
                f"{self.specialist.value} specialist exceeded its "
                f"{self.max_steps}-step budget"
            )

    def require_action(self, action: ForbiddenSpecialistAction) -> None:
        if action in self.forbidden_actions:
            raise SpecialistPermissionDenied(
                f"{self.specialist.value} specialist cannot perform action: "
                f"{action.value}"
            )


_COMMON_FORBIDDEN_ACTIONS = frozenset(
    {
        ForbiddenSpecialistAction.REAL_WORLD_SEND,
        ForbiddenSpecialistAction.RAW_PII_OUTPUT,
        ForbiddenSpecialistAction.CHILD_DIAGNOSIS,
        ForbiddenSpecialistAction.MEDICAL_ADVICE,
        ForbiddenSpecialistAction.LEGAL_COMPLIANCE_CONCLUSION,
        ForbiddenSpecialistAction.UNAPPROVED_WRITE,
    }
)


DEFAULT_SPECIALIST_PERMISSIONS: Mapping[
    SpecialistKind,
    SpecialistPermissionPolicy,
] = MappingProxyType(
    {
        SpecialistKind.PLANNING: SpecialistPermissionPolicy(
            specialist=SpecialistKind.PLANNING,
            allowed_tool_names=frozenset(
                {
                    "load_skill",
                    "get_class_profile",
                    "retrieve_risk_guidance",
                    "check_activity_safety",
                    "align_to_eylf_outcomes",
                    "save_draft",
                    "recall_long_term_memory",
                }
            ),
            max_steps=7,
            forbidden_actions=_COMMON_FORBIDDEN_ACTIONS,
        ),
        SpecialistKind.DOCUMENTATION: SpecialistPermissionPolicy(
            specialist=SpecialistKind.DOCUMENTATION,
            allowed_tool_names=frozenset(),
            max_steps=1,
            forbidden_actions=_COMMON_FORBIDDEN_ACTIONS,
        ),
        SpecialistKind.POLICY: SpecialistPermissionPolicy(
            specialist=SpecialistKind.POLICY,
            allowed_tool_names=frozenset(),
            max_steps=1,
            forbidden_actions=_COMMON_FORBIDDEN_ACTIONS,
        ),
        SpecialistKind.FAMILY: SpecialistPermissionPolicy(
            specialist=SpecialistKind.FAMILY,
            allowed_tool_names=frozenset(),
            max_steps=1,
            forbidden_actions=_COMMON_FORBIDDEN_ACTIONS,
        ),
    }
)


def get_specialist_permission(
    specialist: SpecialistKind,
) -> SpecialistPermissionPolicy:
    """Return the default permission boundary for a specialist."""

    return DEFAULT_SPECIALIST_PERMISSIONS[specialist]
