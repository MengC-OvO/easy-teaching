from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import ApiRuntime, build_api_runtime
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


RuntimeFactory = Callable[[], ApiRuntime]
WEB_DIRECTORY = Path(__file__).resolve().parent / "web"


def create_app(runtime_factory: RuntimeFactory = build_api_runtime) -> FastAPI:
    """Create a FastAPI app with one runtime for its full lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = runtime_factory()
        app.state.runtime = runtime
        recover_incomplete_runs(runtime)
        try:
            yield
        finally:
            runtime.close()

    application = FastAPI(
        title="EduFlow AU Agent",
        description=(
            "Teacher workflow agent backend for synthetic Australian early "
            "childhood education scenarios."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(sessions_router)
    application.include_router(auth_router)
    application.include_router(messages_router)
    application.include_router(drafts_router)
    application.include_router(approvals_router)
    application.include_router(events_router)
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
