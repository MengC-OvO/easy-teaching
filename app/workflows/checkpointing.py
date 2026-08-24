import json
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import set_json_dumps

async def build_postgres_checkpointer(database_url: str) -> AsyncPostgresSaver:
    """Create and initialize the production PostgreSQL checkpointer."""
    connection = await AsyncConnection.connect(
        database_url,
        autocommit=True,
        row_factory=dict_row,
    )
    set_json_dumps(
        lambda value: json.dumps(value, default=_checkpoint_json_default),
        connection,
    )
    checkpointer = AsyncPostgresSaver(connection)
    try:
        await checkpointer.setup()
    except Exception:
        await connection.close()
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
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def checkpoint_config(thread_id: str) -> Dict[str, Dict[str, Any]]:
    return {"configurable": {"thread_id": thread_id}}
