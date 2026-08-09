from typing import Any, List, Optional

from app.tools.controlled_tools.align_to_eylf_outcomes import (
    EylfAlignmentModelProvider,
    build_align_to_eylf_outcomes_tool,
)
from app.tools.controlled_tools.retrieve_risk_guidance import (
    KnowledgeRetrieverProtocol,
    RiskGuidanceModelProvider,
    build_retrieve_risk_guidance_tool,
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
    risk_guidance_model_provider: Optional[RiskGuidanceModelProvider] = None,
    eylf_alignment_model_provider: Optional[EylfAlignmentModelProvider] = None,
) -> List[ToolDefinition]:
    domain_tools = [
        build_get_class_profile_tool(store),
        build_retrieve_risk_guidance_tool(
            knowledge_retriever,
            model_provider=risk_guidance_model_provider,
        ),
        build_check_activity_safety_tool(),
        build_align_to_eylf_outcomes_tool(
            knowledge_retriever,
            model_provider=eylf_alignment_model_provider,
        ),
        build_recall_long_term_memory_tool(store),
        build_search_public_resources_tool(),
        build_get_public_weather_tool(),
    ]
    return domain_tools


def build_default_tool_registry(
    store: Any,
    *,
    knowledge_retriever: Optional[KnowledgeRetrieverProtocol] = None,
    risk_guidance_model_provider: Optional[RiskGuidanceModelProvider] = None,
    eylf_alignment_model_provider: Optional[EylfAlignmentModelProvider] = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in build_default_tool_definitions(
        store,
        knowledge_retriever=knowledge_retriever,
        risk_guidance_model_provider=risk_guidance_model_provider,
        eylf_alignment_model_provider=eylf_alignment_model_provider,
    ):
        registry.register(tool)
    return registry
