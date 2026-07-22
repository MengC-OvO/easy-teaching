import sqlite3
from pathlib import Path
from typing import Any, Dict, Union

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings


CheckpointPath = Union[str, Path]


def build_sqlite_checkpointer(
    database_path: CheckpointPath = settings.checkpoint_database_path,
) -> SqliteSaver:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(connection)


def checkpoint_config(thread_id: str) -> Dict[str, Dict[str, Any]]:
    return {"configurable": {"thread_id": thread_id}}
