"""Application-scoped resources used by future EduFlow API endpoints."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config import settings
from app.services import EduFlowStore
from app.workflows import (
    build_main_graph,
    build_postgres_checkpointer,
    build_sqlite_checkpointer,
)


RuntimePath = Union[str, Path]


@dataclass
class ApiRuntime:
    """Own the shared store, checkpointer, and compiled graph for one app."""

    store: EduFlowStore
    checkpointer: BaseCheckpointSaver
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
            connection = getattr(self.checkpointer, "conn", None)
            if connection is not None:
                connection.close()
        finally:
            self.store.engine.dispose()
            self._closed = True


def build_api_runtime(
    *,
    database_url: Optional[str] = None,
    checkpoint_database_url: Optional[str] = None,
    database_path: Optional[RuntimePath] = None,
    checkpoint_database_path: Optional[RuntimePath] = None,
) -> ApiRuntime:
    """Build the application runtime around the existing workflow components."""
    use_sqlite_store = database_path is not None
    resolved_database_url = (
        f"sqlite:///{Path(database_path)}"
        if use_sqlite_store
        else database_url or settings.database_url
    )
    if not resolved_database_url:
        resolved_database_url = f"sqlite:///{Path(settings.database_path)}"
        use_sqlite_store = True

    use_sqlite_checkpointer = checkpoint_database_path is not None
    resolved_checkpoint_url = (
        checkpoint_database_url or settings.checkpoint_database_url
    )

    store = EduFlowStore(resolved_database_url)
    try:
        store.initialize(create_schema=use_sqlite_store)
        if use_sqlite_checkpointer or not resolved_checkpoint_url:
            checkpoint_path = (
                checkpoint_database_path
                if checkpoint_database_path is not None
                else settings.checkpoint_database_path
            )
            checkpointer = build_sqlite_checkpointer(checkpoint_path)
        else:
            checkpointer = build_postgres_checkpointer(resolved_checkpoint_url)
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
