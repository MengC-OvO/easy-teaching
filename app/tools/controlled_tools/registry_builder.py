from typing import Any, List, Optional

from app.tools.controlled_tools.knowledge_search import (
    KnowledgeRerankerProtocol,
    KnowledgeRetrieverProtocol,
    QueryRewriteModelProvider,
    build_retrieve_knowledge_tool,
)
from app.tools.controlled_tools.check_activity_safety import build_check_activity_safety_tool
from app.tools.controlled_tools.class_context import build_get_class_context_tool
from app.tools.controlled_tools.daily_context import build_get_daily_context_tool
from app.tools.controlled_tools.draft_artifacts import build_read_draft_artifact_tool
from app.tools.controlled_tools.export_records import build_export_records_tool
from app.tools.controlled_tools.records import (
    build_query_records_tool,
    build_save_educational_record_tool,
    build_save_observation_tool,
)
from app.tools.controlled_tools.google_drive import build_google_drive_tools
from app.tools.definition import ToolDefinition
from app.tools.mcp_adapter import MCPClientProtocol
from app.tools.registry import ToolRegistry


def build_default_tool_definitions(
    store: Any,
    *,
    knowledge_retriever: Optional[KnowledgeRetrieverProtocol] = None,
    query_rewriter: Optional[QueryRewriteModelProvider] = None,
    knowledge_reranker: Optional[KnowledgeRerankerProtocol] = None,
    google_drive_mcp_client: Optional[MCPClientProtocol] = None,
    google_drive_user_email: str = "",
    google_drive_mcp_timeout_seconds: float = 45.0,
) -> List[ToolDefinition]:
    domain_tools = [
        build_get_class_context_tool(store),
        build_retrieve_knowledge_tool(
            knowledge_retriever,
            query_rewriter=query_rewriter,
            reranker=knowledge_reranker,
        ),
        build_query_records_tool(store),
        build_read_draft_artifact_tool(store),
        build_get_daily_context_tool(store),
        build_check_activity_safety_tool(),
        build_save_observation_tool(store),
        build_save_educational_record_tool(store),
        build_export_records_tool(store),
    ]
    if google_drive_mcp_client is not None:
        if not google_drive_user_email:
            raise ValueError("GOOGLE_DRIVE_USER_EMAIL is required when Drive MCP is enabled")
        domain_tools.extend(
            build_google_drive_tools(
                store,
                client=google_drive_mcp_client,
                user_google_email=google_drive_user_email,
                timeout_seconds=google_drive_mcp_timeout_seconds,
            )
        )
    return domain_tools


def build_default_tool_registry(
    store: Any,
    *,
    knowledge_retriever: Optional[KnowledgeRetrieverProtocol] = None,
    query_rewriter: Optional[QueryRewriteModelProvider] = None,
    knowledge_reranker: Optional[KnowledgeRerankerProtocol] = None,
    google_drive_mcp_client: Optional[MCPClientProtocol] = None,
    google_drive_user_email: str = "",
    google_drive_mcp_timeout_seconds: float = 45.0,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_default_tool_definitions(
        store,
        knowledge_retriever=knowledge_retriever,
        query_rewriter=query_rewriter,
        knowledge_reranker=knowledge_reranker,
        google_drive_mcp_client=google_drive_mcp_client,
        google_drive_user_email=google_drive_user_email,
        google_drive_mcp_timeout_seconds=google_drive_mcp_timeout_seconds,
    ):
        registry.register(tool)
    return registry
