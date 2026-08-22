"""Versioned, strict HTTP contracts shared by EasyTeaching and its gateway."""
from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


CONTRACT_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputSource(str, Enum):
    USER_MESSAGE = "user_message"
    USER_FORM = "user_form"
    VOICE_TRANSCRIPT = "voice_transcript"


class GatewayAction(str, Enum):
    ALLOW = "allow"
    CLARIFY = "clarify"
    BLOCK = "block"


class InjectionRisk(str, Enum):
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    BLOCK = "block"


class EducationScope(str, Enum):
    IN_SCOPE = "in_scope"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


class ProfessionalRisk(str, Enum):
    NONE = "none"
    MEDICAL = "medical"
    LEGAL = "legal"
    SAFEGUARDING = "safeguarding"


class EntityLabel(str, Enum):
    PERSON_NAME = "PERSON_NAME"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    DOB = "DOB"


class InspectRequest(StrictModel):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: UUID
    session_id: UUID | None = None
    source: InputSource = InputSource.USER_MESSAGE
    text: str = Field(min_length=1, max_length=20_000, strict=True)


class SafetySignals(StrictModel):
    injection_risk: InjectionRisk
    education_scope: EducationScope
    professional_risk: ProfessionalRisk


class InspectResponse(StrictModel):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: UUID
    action: GatewayAction
    reason_code: str = Field(min_length=1, max_length=80, strict=True)
    signals: SafetySignals
    redacted_text: str | None = Field(default=None, max_length=30_000)
    mapping_id: str | None = Field(default=None, min_length=1, max_length=160)
    entity_counts: dict[EntityLabel, int] = Field(default_factory=dict)


class GatewayHealth(StrictModel):
    status: Literal["ok"] = "ok"
    service: Literal["easyteaching-local-safety-gateway"] = "easyteaching-local-safety-gateway"
    contract_version: Literal["1.0"] = CONTRACT_VERSION


class GatewayReadiness(StrictModel):
    ready: bool
    model_loaded: bool
    contract_version: Literal["1.0"] = CONTRACT_VERSION


class GatewayError(StrictModel):
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    request_id: UUID
    error_code: str = Field(min_length=1, max_length=80, strict=True)
    message: str = Field(min_length=1, max_length=200, strict=True)
