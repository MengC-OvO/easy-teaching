"""Application-scoped resources used by future EasyTeaching API endpoints."""

from dataclasses import dataclass, field
from pathlib import Path
import os
import shutil
import sys
from typing import Any, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings
from app.integrations.privacy_gateway_client import PrivacyGatewayClient
from app.services import AsyncEasyTeachingStore
from app.workflows import (
    build_main_react_graph,
    build_postgres_checkpointer,
)
from app.tools import StdioMCPClient, ToolRegistry, build_default_tool_registry


@dataclass
class ApiRuntime:
    """Own the shared store, checkpointer, and compiled graph for one app."""

    store: AsyncEasyTeachingStore
    checkpointer: AsyncPostgresSaver
    graph: Any
    tool_registry: ToolRegistry
    privacy_gateway_mode: str = "disabled"
    privacy_gateway_client: Optional[PrivacyGatewayClient] = None
    google_drive_mcp_client: Optional[StdioMCPClient] = None
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
                try:
                    if self.google_drive_mcp_client is not None:
                        await self.google_drive_mcp_client.aclose()
                finally:
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
    """Build the application runtime around the Main ReAct agent graph."""
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

    google_drive_mcp_client = None
    if settings.google_drive_mcp_enabled:
        command = settings.google_drive_mcp_command
        if command == "workspace-mcp" and shutil.which(command) is None:
            executable_name = (
                "workspace-mcp.exe" if sys.platform == "win32" else "workspace-mcp"
            )
            local_command = Path(sys.executable).with_name(executable_name)
            if local_command.exists():
                command = str(local_command)
        google_drive_mcp_client = StdioMCPClient(
            command=command,
            args=[
                "--single-user",
                "--tools",
                "drive",
                "--tool-tier",
                "core",
                "--transport",
                "stdio",
            ],
            env={
                **os.environ,
                "GOOGLE_OAUTH_CLIENT_ID": settings.google_oauth_client_id,
                "GOOGLE_OAUTH_CLIENT_SECRET": settings.google_oauth_client_secret,
                "WORKSPACE_MCP_CREDENTIALS_DIR": str(
                    Path(settings.google_workspace_mcp_credentials_dir).resolve()
                ),
                "WORKSPACE_MCP_LOG_DIR": str(
                    Path("data/local/google_workspace_mcp/logs").resolve()
                ),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )

    try:
        tool_registry = build_default_tool_registry(
            store,
            google_drive_mcp_client=google_drive_mcp_client,
            google_drive_user_email=settings.google_drive_user_email,
            google_drive_mcp_timeout_seconds=(
                settings.google_drive_mcp_timeout_seconds
            ),
        )
        graph = build_main_react_graph(
            checkpointer=checkpointer,
            long_memory_store=store,
            registry=tool_registry,
        )
    except Exception:
        if google_drive_mcp_client is not None:
            await google_drive_mcp_client.aclose()
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
        tool_registry=tool_registry,
        privacy_gateway_mode=settings.privacy_gateway_mode,
        privacy_gateway_client=privacy_gateway_client,
        google_drive_mcp_client=google_drive_mcp_client,
    )
