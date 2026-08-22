"""FastAPI runtime exports loaded lazily to keep route imports lightweight."""
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.api.runtime import ApiRuntime, build_api_runtime

__all__ = ["ApiRuntime", "build_api_runtime"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    runtime_module = import_module("app.api.runtime")
    value = getattr(runtime_module, name)
    globals()[name] = value
    return value
