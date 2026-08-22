"""Rules + local-model + deterministic-redaction orchestration."""
from __future__ import annotations

from collections import Counter
from typing import Protocol

from safety_gateway.contracts import (
    EducationScope,
    GatewayAction,
    InjectionRisk,
    InspectRequest,
    InspectResponse,
    ModelAnnotation,
    ProfessionalRisk,
    RestoreResponse,
    SafetySignals,
)
from safety_gateway.model import SafetyAnnotator
from safety_gateway.redaction import premask_text, resolve_redaction, rule_injection_risk
from safety_gateway.vault import InMemoryMappingVault, MappingNotFoundError


class GatewayNotReadyError(RuntimeError):
    """Raised without including raw request text."""


class GatewayProcessingError(RuntimeError):
    """Fail-closed processing error with no source text."""


class GatewayMappingNotFoundError(LookupError):
    """Opaque mapping id is unavailable or has expired."""


class SafetyPipeline(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_loaded(self) -> bool: ...

    async def inspect(self, request: InspectRequest) -> InspectResponse: ...

    async def restore(self, mapping_id: str, text: str) -> RestoreResponse: ...

    async def discard(self, mapping_id: str) -> None: ...


class UnavailableSafetyPipeline:
    """Safe default: liveness succeeds, readiness and inspection fail closed."""

    ready = False
    model_loaded = False

    async def inspect(self, request: InspectRequest) -> InspectResponse:
        del request
        raise GatewayNotReadyError("Local safety pipeline is not ready")

    async def restore(self, mapping_id: str, text: str) -> RestoreResponse:
        del mapping_id, text
        raise GatewayNotReadyError("Local safety pipeline is not ready")

    async def discard(self, mapping_id: str) -> None:
        del mapping_id
        raise GatewayNotReadyError("Local safety pipeline is not ready")


RISK_ORDER = {
    InjectionRisk.NORMAL: 0,
    InjectionRisk.SUSPICIOUS: 1,
    InjectionRisk.BLOCK: 2,
}


def policy_action(annotation: ModelAnnotation) -> tuple[GatewayAction, str]:
    if annotation.injection_risk is InjectionRisk.BLOCK:
        return GatewayAction.BLOCK, "prompt_injection"
    if annotation.injection_risk is InjectionRisk.SUSPICIOUS:
        return GatewayAction.CLARIFY, "suspicious_prompt_review"
    if annotation.professional_risk is ProfessionalRisk.MEDICAL:
        return GatewayAction.BLOCK, "medical_boundary"
    if annotation.professional_risk is ProfessionalRisk.LEGAL:
        return GatewayAction.BLOCK, "legal_boundary"
    if annotation.professional_risk is ProfessionalRisk.SAFEGUARDING:
        return GatewayAction.BLOCK, "safeguarding_escalation"
    if annotation.education_scope is EducationScope.OUT_OF_SCOPE:
        return GatewayAction.BLOCK, "outside_education_scope"
    if annotation.education_scope is EducationScope.AMBIGUOUS:
        return GatewayAction.CLARIFY, "ambiguous_education_scope"
    return GatewayAction.ALLOW, "allowed"


class LocalSafetyPipeline:
    """The model annotates; Python owns rules, policy, redaction, and mappings."""

    def __init__(self, *, annotator: SafetyAnnotator, vault: InMemoryMappingVault) -> None:
        self._annotator = annotator
        self._vault = vault

    @property
    def ready(self) -> bool:
        return self._annotator.ready

    @property
    def model_loaded(self) -> bool:
        return self._annotator.model_loaded

    async def inspect(self, request: InspectRequest) -> InspectResponse:
        try:
            premasked = premask_text(request.text)
            annotation = await self._annotator.annotate(premasked.text)
            rule_risk = rule_injection_risk(request.text)
            if RISK_ORDER[rule_risk] > RISK_ORDER[annotation.injection_risk]:
                annotation = annotation.model_copy(update={"injection_risk": rule_risk})
            redaction = resolve_redaction(request.text, premasked, annotation)
            action, reason_code = policy_action(annotation)
            mapping_id = None
            redacted_text = None
            if action is GatewayAction.ALLOW:
                redacted_text = redaction.text
                mapping_id = await self._vault.save(redaction.mappings)
            return InspectResponse(
                request_id=request.request_id,
                action=action,
                reason_code=reason_code,
                signals=SafetySignals(
                    injection_risk=annotation.injection_risk,
                    education_scope=annotation.education_scope,
                    professional_risk=annotation.professional_risk,
                ),
                redacted_text=redacted_text,
                mapping_id=mapping_id,
                entity_counts=dict(Counter(mapping.label for mapping in redaction.mappings)),
            )
        except GatewayNotReadyError:
            raise
        except Exception as error:
            raise GatewayProcessingError("local safety inspection failed") from error

    async def restore(self, mapping_id: str, text: str) -> RestoreResponse:
        try:
            restored = await self._vault.consume_and_restore(mapping_id, text)
            return RestoreResponse(restored_text=restored)
        except MappingNotFoundError as error:
            raise GatewayMappingNotFoundError("mapping is unavailable") from error

    async def discard(self, mapping_id: str) -> None:
        await self._vault.discard(mapping_id)
