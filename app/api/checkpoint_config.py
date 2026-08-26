"""Lightweight LangGraph invocation configuration shared by API orchestration."""
from typing import Any, Dict


def checkpoint_config(thread_id: str) -> Dict[str, Any]:
    # One ReAct turn spans several LangGraph nodes. Keep LangGraph's own limit
    # above the graph's explicit max_steps so the bounded fallback runs first.
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
