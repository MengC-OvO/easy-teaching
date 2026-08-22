import asyncio
from uuid import uuid4

import httpx

from app.integrations.privacy_gateway_client import PrivacyGatewayClient
from services.local_safety_gateway.contracts import InspectRequest


def test_client_serializes_request_and_validates_response() -> None:
    request_id = uuid4()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/inspect"
        return httpx.Response(
            200,
            json={
                "contract_version": "1.0",
                "request_id": str(request_id),
                "action": "clarify",
                "reason_code": "synthetic_test",
                "signals": {
                    "injection_risk": "suspicious",
                    "education_scope": "in_scope",
                    "professional_risk": "none",
                },
                "redacted_text": None,
                "mapping_id": None,
                "entity_counts": {},
            },
        )

    async def run() -> None:
        async with httpx.AsyncClient(
            base_url="http://gateway.test",
            transport=httpx.MockTransport(handler),
        ) as http_client:
            client = PrivacyGatewayClient(
                base_url="http://gateway.test",
                timeout_seconds=1,
                http_client=http_client,
            )
            result = await client.inspect(
                InspectRequest(request_id=request_id, text="Synthetic suspicious quote")
            )
            assert result.action.value == "clarify"
            assert result.signals.injection_risk.value == "suspicious"

    asyncio.run(run())
