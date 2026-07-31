"""Application-scoped resources used by future EduFlow API endpoints."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings
from app.services import EduFlowStore
from app.workflows import build_main_graph, build_sqlite_checkpointer


RuntimePath = Union[str, Path]


@dataclass
class ApiRuntime:
    """Own the shared store, checkpointer, and compiled graph for one app."""

    store: EduFlowStore
    checkpointer: SqliteSaver
    graph: Any
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Release runtime-owned resources exactly once."""
        if self._closed:
            return

        try:
            self.checkpointer.conn.close()
        finally:
            self.store.engine.dispose()
            self._closed = True


def build_api_runtime(
    *,
    database_path: RuntimePath = settings.database_path,
    checkpoint_database_path: RuntimePath = settings.checkpoint_database_path,
) -> ApiRuntime:
    """Build the application runtime around the existing workflow components."""
    store = EduFlowStore(f"sqlite:///{Path(database_path)}")
    try:
        store.initialize()
        checkpointer = build_sqlite_checkpointer(checkpoint_database_path)
    except Exception:
        store.engine.dispose()
        raise

    try:
        graph = build_main_graph(
            checkpointer=checkpointer,
            long_memory_store=store,
            learning_record_store=store,
        )
    except Exception:
        checkpointer.conn.close()
        store.engine.dispose()
        raise

    return ApiRuntime(
        store=store,
        checkpointer=checkpointer,
        graph=graph,
    )
