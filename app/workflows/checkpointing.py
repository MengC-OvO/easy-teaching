import sqlite3
import json
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union
from uuid import UUID

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel
from psycopg import connect
from psycopg.rows import dict_row
from psycopg.types.json import set_json_dumps

from app.config import settings


CheckpointPath = Union[str, Path]


def build_sqlite_checkpointer(
    database_path: CheckpointPath = settings.checkpoint_database_path,
) -> SqliteSaver:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection)


def build_postgres_checkpointer(database_url: str) -> PostgresSaver:
    """Create and initialize the production PostgreSQL checkpointer."""
    connection = connect(
        database_url,
        autocommit=True,
        row_factory=dict_row,
    )
    set_json_dumps(
        lambda value: json.dumps(value, default=_checkpoint_json_default),
        connection,
    )
    checkpointer = PostgresSaver(connection)
    try:
        checkpointer.setup()
    except Exception:
        connection.close()
        raise
    return checkpointer


def _checkpoint_json_default(value: Any) -> Any:
    """Encode validated graph values for PostgresSaver JSONB metadata."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, (Path, UUID)):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def checkpoint_config(thread_id: str) -> Dict[str, Dict[str, Any]]:
    return {"configurable": {"thread_id": thread_id}}
