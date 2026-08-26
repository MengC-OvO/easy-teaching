from contextlib import asynccontextmanager
from pathlib import Path
import inspect
from typing import Any, Awaitable, Callable, Optional, Union

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.recovery import recover_incomplete_runs
from app.api.routes import (
    approvals_router,
    drafts_router,
    events_router,
    messages_router,
    sessions_router,
)
from app.config import settings


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
        await recover_incomplete_runs(runtime)
        try:
            yield
        finally:
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

    return application


app = create_app()
