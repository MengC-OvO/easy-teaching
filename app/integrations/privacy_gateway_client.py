"""Typed HTTP client for the local Privacy & Safety Gateway."""
from __future__ import annotations

from typing import Optional

import httpx
from pydantic import ValidationError

from services.local_safety_gateway.contracts import InspectRequest, InspectResponse


class PrivacyGatewayError(RuntimeError):
    """Base error that contains no raw user text."""


class PrivacyGatewayUnavailableError(PrivacyGatewayError):
    """The local gateway could not provide a valid response."""


class PrivacyGatewayClient:
    """Call the gateway without logging or embedding the request text in errors."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    async def inspect(self, request: InspectRequest) -> InspectResponse:
        try:
            response = await self._client.post(
                "/v1/inspect",
                content=request.model_dump_json(),
                headers={"content-type": "application/json"},
            )
        except httpx.HTTPError as error:
            raise PrivacyGatewayUnavailableError(
                "Local privacy gateway request failed"
            ) from error

        if response.status_code != 200:
            raise PrivacyGatewayUnavailableError(
                f"Local privacy gateway returned HTTP {response.status_code}"
            )
        try:
            return InspectResponse.model_validate_json(response.content)
        except ValidationError as error:
            raise PrivacyGatewayUnavailableError(
                "Local privacy gateway returned an invalid contract"
            ) from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
