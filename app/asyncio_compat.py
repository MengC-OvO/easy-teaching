"""Cross-platform helpers for running async entry points.

Psycopg's async driver requires a selector-based event loop on Windows.
"""

import asyncio
import selectors
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


def _event_loop_factory() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def run_async(coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a top-level coroutine with an event loop supported by psycopg."""

    with asyncio.Runner(loop_factory=_event_loop_factory) as runner:
        return runner.run(coroutine)
