"""FastAPI boundary for the local Privacy & Safety Gateway."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from services.local_safety_gateway.contracts import (
    CONTRACT_VERSION,
    GatewayError,
    GatewayHealth,
    GatewayReadiness,
    InspectRequest,
    InspectResponse,
)
from services.local_safety_gateway.pipeline import (
    GatewayNotReadyError,
    SafetyPipeline,
    UnavailableSafetyPipeline,
)


def create_app(pipeline: SafetyPipeline | None = None) -> FastAPI:
    application = FastAPI(
        title="EasyTeaching Local Privacy & Safety Gateway",
        version=CONTRACT_VERSION,
        description="Local-only input annotation and deterministic safety boundary.",
    )
    application.state.pipeline = pipeline or UnavailableSafetyPipeline()

    @application.get("/health", response_model=GatewayHealth)
    def health() -> GatewayHealth:
        return GatewayHealth()

    @application.get(
        "/ready",
        response_model=GatewayReadiness,
        responses={503: {"model": GatewayReadiness}},
    )
    def ready():
        result = GatewayReadiness(
            ready=application.state.pipeline.ready,
            model_loaded=application.state.pipeline.model_loaded,
        )
        if not result.ready:
            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        return result

    @application.post(
        "/v1/inspect",
        response_model=InspectResponse,
        responses={503: {"model": GatewayError}},
    )
    async def inspect(request: InspectRequest):
        try:
            return await application.state.pipeline.inspect(request)
        except GatewayNotReadyError:
            error = GatewayError(
                request_id=request.request_id,
                error_code="gateway_not_ready",
                message="Local safety pipeline is not ready",
            )
            return JSONResponse(status_code=503, content=error.model_dump(mode="json"))

    return application


app = create_app()
