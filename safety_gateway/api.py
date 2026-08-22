"""FastAPI boundary for the local Privacy & Safety Gateway."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from safety_gateway.contracts import (
    CONTRACT_VERSION,
    GatewayError,
    GatewayHealth,
    GatewayReadiness,
    InspectRequest,
    InspectResponse,
    RestoreRequest,
    RestoreResponse,
)
from safety_gateway.pipeline import (
    GatewayMappingNotFoundError,
    GatewayNotReadyError,
    GatewayProcessingError,
    SafetyPipeline,
    UnavailableSafetyPipeline,
)


def create_app(
    pipeline: SafetyPipeline | None = None,
    lifespan: Callable[[FastAPI], AsyncIterator[None]] | None = None,
) -> FastAPI:
    application = FastAPI(
        title="EasyTeaching Local Privacy & Safety Gateway",
        version=CONTRACT_VERSION,
        description="Local-only input annotation and deterministic safety boundary.",
        lifespan=lifespan,
    )
    application.state.pipeline = pipeline or UnavailableSafetyPipeline()

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_request, _error: RequestValidationError):
        error = GatewayError(
            error_code="invalid_request",
            message="Request does not match the safety gateway contract",
        )
        return JSONResponse(status_code=422, content=error.model_dump(mode="json"))

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
        except GatewayProcessingError:
            error = GatewayError(
                request_id=request.request_id,
                error_code="inspection_failed",
                message="Local safety inspection failed closed",
            )
            return JSONResponse(status_code=503, content=error.model_dump(mode="json"))

    @application.post(
        "/v1/restore",
        response_model=RestoreResponse,
        responses={404: {"model": GatewayError}, 503: {"model": GatewayError}},
    )
    async def restore(request: RestoreRequest):
        try:
            return await application.state.pipeline.restore(request.mapping_id, request.text)
        except GatewayNotReadyError:
            error = GatewayError(
                error_code="gateway_not_ready",
                message="Local safety pipeline is not ready",
            )
            return JSONResponse(status_code=503, content=error.model_dump(mode="json"))
        except GatewayMappingNotFoundError:
            error = GatewayError(
                error_code="mapping_unavailable",
                message="Mapping is expired, consumed, or unknown",
            )
            return JSONResponse(status_code=404, content=error.model_dump(mode="json"))

    @application.delete(
        "/v1/mappings/{mapping_id}",
        status_code=204,
        responses={503: {"model": GatewayError}},
    )
    async def discard(mapping_id: str):
        try:
            await application.state.pipeline.discard(mapping_id)
            return Response(status_code=204)
        except GatewayNotReadyError:
            error = GatewayError(
                error_code="gateway_not_ready",
                message="Local safety pipeline is not ready",
            )
            return JSONResponse(status_code=503, content=error.model_dump(mode="json"))

    return application


app = create_app()
