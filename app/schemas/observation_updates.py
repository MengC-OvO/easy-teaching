"""Deterministic delta/replacement protocol for older LangGraph reducers."""
from typing import Any, Mapping

REPLACE = "__replace_observations__"


def ordered(items):
    positions = {name: index for index, name in enumerate(items)}
    def key(pair):
        name, value = pair
        get = value.get if isinstance(value, dict) else lambda field, default=None: getattr(value, field, default)
        batch = get("batch_id")
        if not batch:
            return (-1, positions[name], name)
        return (int(batch.rsplit(":", 1)[-1]), get("call_index", 0), name)
    return dict(sorted(items.items(), key=key))


def replace_observations(items: Mapping[str, Any]) -> dict:
    return {REPLACE: dict(items)}


def merge_observation_updates(old: dict, incoming: dict) -> dict:
    if REPLACE in incoming:
        if len(incoming) != 1 or not isinstance(incoming[REPLACE], dict):
            raise ValueError("Invalid observation replacement")
        return ordered(incoming[REPLACE])
    merged = dict(old or {})
    for key, value in incoming.items():
        if key == REPLACE:
            raise ValueError("Reserved observation key")
        previous = merged.get(key)
        serialize = lambda item: item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        if previous is not None and serialize(previous) != serialize(value):
            raise ValueError(f"Conflicting observation result_key: {key}")
        merged[key] = value
    return ordered(merged)
