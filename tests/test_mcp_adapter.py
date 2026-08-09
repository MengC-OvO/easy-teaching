from pydantic import BaseModel

from app.agents import MainToolExecutor
from app.schemas import CapabilityCall, CapabilitySource
from app.tools import ToolRegistry, build_read_only_mcp_tool


class MCPInput(BaseModel):
    query: str


class MCPOutput(BaseModel):
    answer: str


class StubMCPClient:
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
