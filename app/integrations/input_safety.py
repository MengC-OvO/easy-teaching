"""Mode-aware preparation of one user message before persistence or ReAct."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

from app.integrations.privacy_gateway_client import (
    PrivacyGatewayClient,
    PrivacyGatewayUnavailableError,
)
from safety_gateway.contracts import GatewayAction, InspectRequest, InspectResponse


GatewayMode = Literal["disabled", "shadow", "enforce"]


@dataclass(frozen=True)
class PreparedUserInput:
    """Text safe to forward plus the opaque id needed for later restoration."""

    forwarded_text: str
    mapping_id: str | None
    inspection: InspectResponse | None


class InputSafetyRejected(RuntimeError):
    """A valid gateway decision stopped the request without retaining source text."""

    def __init__(self, inspection: InspectResponse) -> None:
        super().__init__(inspection.reason_code)
        self.inspection = inspection


def _optional_uuid(value: str) -> UUID | None:
    try:
        return UUID(value)
    except ValueError:
        return None


async def prepare_user_input(
    *,
    mode: GatewayMode,
    client: PrivacyGatewayClient | None,
    session_id: str,
    text: str,
) -> PreparedUserInput:
    """Inspect before persistence; shadow never changes the forwarded text."""
    if mode == "disabled":
        return PreparedUserInput(text, None, None)
    if client is None:
        raise PrivacyGatewayUnavailableError("Local privacy gateway client is unavailable")

    try:
        inspection = await client.inspect(
            InspectRequest(
                request_id=uuid4(),
                session_id=_optional_uuid(session_id),
                text=text,
            )
        )
    except PrivacyGatewayUnavailableError:
        if mode == "shadow":
            return PreparedUserInput(text, None, None)
        raise

    if mode == "shadow":
        if inspection.mapping_id:
            try:
                await client.discard(inspection.mapping_id)
            except PrivacyGatewayUnavailableError:
                pass
        return PreparedUserInput(text, None, inspection)

    if inspection.action is not GatewayAction.ALLOW:
        raise InputSafetyRejected(inspection)
    if inspection.redacted_text is None:
        raise PrivacyGatewayUnavailableError(
            "Local privacy gateway allowed input without redacted text"
        )
    return PreparedUserInput(
        forwarded_text=inspection.redacted_text,
        mapping_id=inspection.mapping_id,
        inspection=inspection,
    )
