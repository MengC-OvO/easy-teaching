"""Allowlisted official-web search Tool definition."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolResult,
)


class OfficialWebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    domains: List[str] = Field(default_factory=list, max_length=10)
    top_k: int = Field(default=5, ge=1, le=10)


class OfficialWebSearchOutput(BaseModel):
    query: str
    results: List[Dict[str, str]]
    returned_count: int


def build_official_web_search_tool(client: Any) -> ToolDefinition:
    async def run(input_data: BaseModel) -> ToolResult:
        data = OfficialWebSearchInput.model_validate(input_data)
        response = await client.search(
            data.query,
            domains=data.domains or None,
            top_k=data.top_k,
        )
        payload = response.model_dump(mode="json")
        payload["returned_count"] = len(payload["results"])
        return ToolResult.ok(data=payload, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="search_official_web",
        description=(
            "Search current Australian education guidance on configured government "
            "and ACECQA domains only. Use for recent official facts not covered by "
            "the local RAG corpus."
        ),
        category=ToolCategory.WEB,
        input_model=OfficialWebSearchInput,
        output_model=OfficialWebSearchOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.EXTERNAL,
        parallel_safe=True,
        timeout_seconds=15,
        async_handler=run,
    )
