"""Shared FastAPI dependencies for API route modules."""

from typing import TYPE_CHECKING, cast

from fastapi import Request

if TYPE_CHECKING:
    from app.api.runtime import ApiRuntime


def get_runtime(request: Request) -> "ApiRuntime":
    """Return the application-scoped runtime attached by FastAPI lifespan."""
    return cast("ApiRuntime", request.app.state.runtime)

