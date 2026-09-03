from contextlib import asynccontextmanager
from pathlib import Path
import inspect
from typing import Any, Awaitable, Callable, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.recovery import recover_incomplete_runs
from app.api.routes import (
    approvals_router,
    drafts_router,
    events_router,
    messages_router,
    sessions_router,
    uploads_router,
)
from app.config import settings
from app.tasks.dispatcher import OutboxRelay


RuntimeFactory = Callable[[], Union[Any, Awaitable[Any]]]
WEB_DIRECTORY = Path(__file__).resolve().parent / "web"


async def _build_default_runtime():
    from app.api.runtime import build_api_runtime

    return await build_api_runtime()


def create_app(runtime_factory: Optional[RuntimeFactory] = None) -> FastAPI:
    """Create a FastAPI app with one runtime for its full lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_factory = runtime_factory or _build_default_runtime
        runtime_or_awaitable = resolved_factory()
        runtime = (
            await runtime_or_awaitable
            if inspect.isawaitable(runtime_or_awaitable)
            else runtime_or_awaitable
        )
        app.state.runtime = runtime
        outbox_relay = None
        if settings.task_execution_mode == "celery" and hasattr(
            runtime.store, "list_publishable_conversation_task_ids"
        ):
            outbox_relay = OutboxRelay(runtime.store)
            outbox_relay.start()
            app.state.outbox_relay = outbox_relay
            outbox_relay.notify()
        else:
            await recover_incomplete_runs(runtime)
        try:
            yield
        finally:
            if outbox_relay is not None:
                await outbox_relay.close()
            await runtime.close()

    application = FastAPI(
        title="EasyTeaching",
        description=(
            "Safety-aware AI agent backend for synthetic Australian early "
            "childhood education scenarios."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(sessions_router)
    application.include_router(auth_router)
    application.include_router(messages_router)
    application.include_router(drafts_router)
    application.include_router(events_router)
    application.include_router(approvals_router)
    application.include_router(uploads_router)
    application.mount(
        "/assets",
        StaticFiles(directory=WEB_DIRECTORY),
        name="web-assets",
    )

    @application.get("/", include_in_schema=False)
    def web_app() -> FileResponse:
        return FileResponse(WEB_DIRECTORY / "index.html")

    @application.get("/health")
    def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.app_env,
        }

    @application.get("/ready")
    async def readiness_check(request: Request) -> Any:
        runtime = request.app.state.runtime
        checks = {"postgres": "ok", "redis": "ok", "redis_progress": "ok"}
        try:
            if hasattr(runtime.store, "healthcheck"):
                await runtime.store.healthcheck()
        except Exception:
            checks["postgres"] = "unavailable"
        redis_client = getattr(runtime, "redis_client", None)
        if settings.task_execution_mode == "celery" or settings.api_rate_limit_enabled:
            try:
                if redis_client is None or not await redis_client.ping():
                    raise ConnectionError("Redis ping failed")
            except Exception:
                checks["redis"] = "unavailable"
        if settings.task_execution_mode == "celery":
            try:
                progress_client = getattr(runtime, "redis_progress_client", None)
                if progress_client is None or not await progress_client.ping():
                    raise ConnectionError("Redis progress ping failed")
            except Exception:
                checks["redis_progress"] = "unavailable"
        ready = all(value == "ok" for value in checks.values())
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not_ready", "checks": checks},
        )

    return application


app = create_app()
