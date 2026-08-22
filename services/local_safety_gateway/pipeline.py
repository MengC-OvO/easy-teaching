"""Pipeline interface; the real rules/model implementation arrives in step two."""
from __future__ import annotations

from typing import Protocol

from services.local_safety_gateway.contracts import InspectRequest, InspectResponse


class GatewayNotReadyError(RuntimeError):
    """Raised without including raw request text."""


class SafetyPipeline(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_loaded(self) -> bool: ...

    async def inspect(self, request: InspectRequest) -> InspectResponse: ...


class UnavailableSafetyPipeline:
    """Safe default: liveness succeeds, readiness and inspection fail closed."""

    ready = False
    model_loaded = False

    async def inspect(self, request: InspectRequest) -> InspectResponse:
        del request
        raise GatewayNotReadyError("Local safety pipeline is not ready")
