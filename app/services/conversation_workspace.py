"""Metadata-only conversation catalogue, synchronized at lifecycle boundaries."""

import inspect
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.api.checkpoint_config import checkpoint_config

logger = logging.getLogger(__name__)


class ArtifactReference(BaseModel):
    source_request_id: str
    artifact_number: int | None = None
    position_from_latest: int | None = None
    title: str | None = None
    status: str = "unsaved"
    content_chars: int = 0
    created_at: str | None = None


class SavedRecordReference(BaseModel):
    record_id: str | None = None
    source_request_id: str | None = None
    save_number: int | None = None
    tool_name: str | None = None
    record_type: str | None = None
    title: str | None = None
    created_at: str | None = None


class WorkspaceCatalogue(BaseModel):
    # Pydantic ignores unknown fields: raw content/arguments never enter this cache.
    current_artifact: ArtifactReference | None = None
    recent_artifacts: list[ArtifactReference] = Field(default_factory=list, max_length=8)
    recent_saved_record: SavedRecordReference | None = None
    recent_saved_records: list[SavedRecordReference] = Field(default_factory=list, max_length=8)


async def load_workspace(store: Any, *, session_id, teacher_id, class_id):
    reader = getattr(store, "get_conversation_workspace", None)
    if reader is None:
        return None
    value = reader(session_id=session_id, teacher_id=teacher_id, class_id=class_id)
    if inspect.isawaitable(value):
        value = await value
    return WorkspaceCatalogue.model_validate(value).model_dump(mode="json")


async def sync_workspace_checkpoint(runtime, *, session_id: str, request_id: str) -> None:
    """Refresh a finished graph after DB commits without scheduling model work.

    DB remains authoritative. A failed cache refresh is repaired at the next
    initialize; never retry a committed side effect merely to update metadata.
    Call before releasing the active Run to avoid racing the next message.
    """
    updater = getattr(runtime.graph, "aupdate_state", None)
    if updater is None:
        return
    try:
        conversation = await runtime.store.get_conversation_session(session_id)
        if conversation is None:
            return
        config = checkpoint_config(conversation["thread_id"])
        snapshot = await runtime.graph.aget_state(config)
        if (not snapshot.values or snapshot.next
                or snapshot.values.get("request_id") != request_id
                or snapshot.values.get("session_id") != session_id):
            return
        workspace = await load_workspace(
            runtime.store, session_id=session_id,
            teacher_id=conversation.get("teacher_id"),
            class_id=conversation.get("class_id"),
        )
        if workspace is not None:
            await updater(config, {"workspace": workspace}, as_node="long_memory_update")
    except Exception:
        logger.warning("Workspace checkpoint refresh failed; next request will resync",
                       extra={"request_id": request_id}, exc_info=True)
