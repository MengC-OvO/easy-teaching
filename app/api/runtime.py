"""Application-scoped resources used by future EasyTeaching API endpoints."""

from dataclasses import dataclass, field
from typing import Any, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings
from app.integrations.privacy_gateway_client import PrivacyGatewayClient
from app.services import AsyncEasyTeachingStore
from app.workflows import (
    build_main_react_graph,
    build_postgres_checkpointer,
)


@dataclass
class ApiRuntime:
    """Own the shared store, checkpointer, and compiled graph for one app."""

    store: AsyncEasyTeachingStore
    checkpointer: AsyncPostgresSaver
    graph: Any
    privacy_gateway_mode: str = "disabled"
    privacy_gateway_client: Optional[PrivacyGatewayClient] = None
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        """Release runtime-owned resources exactly once."""
        if self._closed:
            return

        try:
            try:
                if self.privacy_gateway_client is not None:
                    await self.privacy_gateway_client.aclose()
            finally:
                connection = getattr(self.checkpointer, "conn", None)
                if connection is not None:
                    await connection.close()
        finally:
            await self.store.close()
            self._closed = True


async def build_api_runtime(
    *,
    database_url: Optional[str] = None,
    checkpoint_database_url: Optional[str] = None,
) -> ApiRuntime:
    """Build the application runtime around the existing workflow components."""
    resolved_database_url = database_url or settings.database_url
    resolved_checkpoint_url = (
        checkpoint_database_url or settings.checkpoint_database_url
    )
    if not resolved_database_url:
        raise ValueError("DATABASE_URL is required for the PostgreSQL runtime")
    if not resolved_checkpoint_url:
        raise ValueError("CHECKPOINT_DATABASE_URL is required for the PostgreSQL runtime")

    store = AsyncEasyTeachingStore(resolved_database_url)
    try:
        await store.initialize()
        checkpointer = await build_postgres_checkpointer(resolved_checkpoint_url)
    except Exception:
        await store.close()
        raise

    try:
        graph = build_main_react_graph(
            checkpointer=checkpointer,
            long_memory_store=store,
        )
    except Exception:
        await checkpointer.conn.close()
        await store.close()
        raise

    privacy_gateway_client = None
    if settings.privacy_gateway_mode != "disabled":
        privacy_gateway_client = PrivacyGatewayClient(
            base_url=settings.privacy_gateway_url,
            timeout_seconds=settings.privacy_gateway_timeout_seconds,
        )

    return ApiRuntime(
        store=store,
        checkpointer=checkpointer,
        graph=graph,
        privacy_gateway_mode=settings.privacy_gateway_mode,
        privacy_gateway_client=privacy_gateway_client,
    )
