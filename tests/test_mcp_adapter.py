from pydantic import BaseModel

from app.agents import MainToolExecutor
from app.schemas import CapabilityCall, CapabilitySource
from app.tools import (
    MCPToolInfo,
    ToolPermission,
    ToolRegistry,
    build_read_only_mcp_tool,
    classify_mcp_tool,
)


class MCPInput(BaseModel):
    query: str


class MCPOutput(BaseModel):
    answer: str


class StubMCPClient:
    async def list_tools(self, **values):
        return []

    async def call_tool(self, *, server_name, tool_name, arguments):
        assert server_name == "approved-public"
        assert tool_name == "search"
        return {"answer": f"Result for {arguments['query']}"}


def test_mcp_adapter_uses_the_same_async_registry_boundary() -> None:
    registry = ToolRegistry()
    registry.register(
        build_read_only_mcp_tool(
            name="public_mcp_search",
            description="Search an approved public MCP source.",
            server_name="approved-public",
            remote_tool_name="search",
            input_model=MCPInput,
            output_model=MCPOutput,
            client=StubMCPClient(),
        )
    )

    import asyncio

    result = asyncio.run(
        registry.execute_async("public_mcp_search", {"query": "play"})
    )

    assert result.success is True
    assert result.data == {"answer": "Result for play"}


def test_main_executor_marks_mcp_observation_source() -> None:
    registry = ToolRegistry()
    registry.register(
        build_read_only_mcp_tool(
            name="public_mcp_search",
            description="Search an approved public MCP source.",
            server_name="approved-public",
            remote_tool_name="search",
            input_model=MCPInput,
            output_model=MCPOutput,
            client=StubMCPClient(),
        )
    )
    executor = MainToolExecutor(registry)

    import asyncio

    observation = asyncio.run(
        executor.execute_one(
            CapabilityCall(
                name="public_mcp_search",
                arguments={"query": "play"},
                result_key="mcp_result",
            ),
            teacher_id=None,
            class_id=None,
        )
    )

    assert observation.source_kind is CapabilitySource.MCP


def test_mcp_risk_classifier_only_trusts_read_only_hint_from_trusted_server():
    tool = MCPToolInfo(
        name="search",
        description="Search",
        input_schema={"type": "object"},
        annotations={"readOnlyHint": True, "destructiveHint": False},
    )

    trusted = classify_mcp_tool(tool, trusted_server=True)
    untrusted = classify_mcp_tool(tool, trusted_server=False)

    assert trusted.permission is ToolPermission.AUTO_EXECUTE
    assert untrusted.permission is ToolPermission.REQUIRE_APPROVAL


def test_mcp_risk_classifier_forbids_destructive_tool():
    tool = MCPToolInfo(
        name="delete_file",
        description="Delete",
        input_schema={"type": "object"},
        annotations={"readOnlyHint": False, "destructiveHint": True},
    )

    decision = classify_mcp_tool(tool, trusted_server=True)

    assert decision.permission is ToolPermission.FORBIDDEN
