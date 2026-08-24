from typing import Any, List, Optional

from app.tools.controlled_tools.knowledge_search import (
    KnowledgeRerankerProtocol,
    KnowledgeRetrieverProtocol,
    QueryRewriteModelProvider,
    build_research_knowledge_tool,
    build_search_knowledge_tool,
)
from app.tools.controlled_tools.check_activity_safety import build_check_activity_safety_tool
from app.tools.controlled_tools.get_class_profile import build_get_class_profile_tool
from app.tools.controlled_tools.recall_long_term_memory import (
    build_recall_long_term_memory_tool,
)
from app.tools.controlled_tools.external_public import (
    build_get_public_weather_tool,
    build_search_public_resources_tool,
)
from app.tools.definition import ToolDefinition
from app.tools.registry import ToolRegistry


def build_default_tool_definitions(
    store: Any,
    *,
    knowledge_retriever: Optional[KnowledgeRetrieverProtocol] = None,
    query_rewriter: Optional[QueryRewriteModelProvider] = None,
    knowledge_reranker: Optional[KnowledgeRerankerProtocol] = None,
) -> List[ToolDefinition]:
    domain_tools = [
        build_get_class_profile_tool(store),
        build_search_knowledge_tool(knowledge_retriever),
        build_research_knowledge_tool(
            knowledge_retriever,
            query_rewriter=query_rewriter,
            reranker=knowledge_reranker,
        ),
        build_check_activity_safety_tool(),
        build_recall_long_term_memory_tool(store),
        build_search_public_resources_tool(),
        build_get_public_weather_tool(),
    ]
    return domain_tools


def build_default_tool_registry(
    store: Any,
    *,
    knowledge_retriever: Optional[KnowledgeRetrieverProtocol] = None,
    query_rewriter: Optional[QueryRewriteModelProvider] = None,
    knowledge_reranker: Optional[KnowledgeRerankerProtocol] = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_default_tool_definitions(
        store,
        knowledge_retriever=knowledge_retriever,
        query_rewriter=query_rewriter,
        knowledge_reranker=knowledge_reranker,
    ):
        registry.register(tool)
    return registry
