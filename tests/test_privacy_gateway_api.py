from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from safety_gateway.api import create_app
from safety_gateway.contracts import (
    GatewayAction,
    InspectRequest,
    InspectResponse,
    SafetySignals,
)


class ReadySyntheticPipeline:
    ready = True
    model_loaded = False

    async def inspect(self, request: InspectRequest) -> InspectResponse:
        return InspectResponse(
            request_id=request.request_id,
            action=GatewayAction.ALLOW,
            reason_code="synthetic_test_stub",
            signals=SafetySignals(
                injection_risk="normal",
                education_scope="in_scope",
                professional_risk="none",
            ),
            redacted_text=request.text,
            entity_counts={},
        )

    async def discard(self, mapping_id: str) -> None:
        self.discarded = mapping_id


def test_default_gateway_is_alive_but_not_ready() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {
        "status": "ok",
        "service": "easyteaching-local-safety-gateway",
        "contract_version": "1.0",
    }
    readiness = client.get("/ready")
    assert readiness.status_code == 503
    assert readiness.json()["ready"] is False


def test_default_inspection_fails_closed_without_echoing_text() -> None:
    client = TestClient(create_app())
    request_id = uuid4()
    private_synthetic_text = "Synthetic child Maya Example, phone 0400 000 001"

    response = client.post(
        "/v1/inspect",
        json={"request_id": str(request_id), "text": private_synthetic_text},
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "gateway_not_ready"
    assert private_synthetic_text not in response.text


def test_injected_pipeline_proves_the_versioned_http_contract() -> None:
    client = TestClient(create_app(ReadySyntheticPipeline()))
    request_id = uuid4()

    response = client.post(
        "/v1/inspect",
        json={"request_id": str(request_id), "text": "Synthetic ECE activity request"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert UUID(payload["request_id"]) == request_id
    assert payload["contract_version"] == "1.0"
    assert payload["action"] == "allow"


def test_discard_endpoint_delegates_without_returning_mapping_data() -> None:
    pipeline = ReadySyntheticPipeline()
    client = TestClient(create_app(pipeline))
    response = client.delete("/v1/mappings/synthetic-opaque-mapping-id")
    assert response.status_code == 204
    assert response.content == b""
    assert pipeline.discarded == "synthetic-opaque-mapping-id"
