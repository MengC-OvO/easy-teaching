"""Execution ownership propagated to database writes, never to model state."""
from contextlib import contextmanager
from contextvars import ContextVar


class LeaseLostError(RuntimeError):
    pass


execution_lease = ContextVar("execution_lease", default=None)


@contextmanager
def bind_execution_lease(request_id: str, token: str):
    handle = execution_lease.set((request_id, token))
    try:
        yield
    finally:
        execution_lease.reset(handle)
