from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from app.api import ApiRuntime, build_api_runtime
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
    application.include_router(messages_router)
    application.include_router(drafts_router)
    application.include_router(approvals_router)
    application.include_router(events_router)

    @application.get("/health")
    def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.app_env,
        }

    return application


app = create_app()
