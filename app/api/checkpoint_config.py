"""Lightweight LangGraph invocation configuration shared by API orchestration."""
from typing import Any, Dict


def checkpoint_config(thread_id: str) -> Dict[str, Dict[str, Any]]:
    return {"configurable": {"thread_id": thread_id}}
