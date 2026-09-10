import asyncio
import copy
from types import SimpleNamespace

from app.tools.dynamic_drive import register_dynamic_drive
from app.tools.mcp_adapter import MCPToolInfo
from app.tools.registry import ToolRegistry
from app.tools.definition import ToolExecutionContext
from app.tools.availability import available_tools_for_state


class Client:
    def __init__(self):
        self.tools = [MCPToolInfo(name="search_drive_files", description="Search Drive files", annotations={"readOnlyHint": True},
            input_schema={"type": "object", "properties": {
                "query": {"type": "string", "minLength": 3, "pattern": "^[a-z]+$"},
                "user_google_email": {"type": "string"}}, "required": ["query", "user_google_email"],
                "additionalProperties": False})]
        self.calls = []

    async def list_tools(self, **kwargs):
        return copy.deepcopy(self.tools)

    async def call_tool(self, **kwargs):
        self.calls.append(kwargs)
        return {"text": "synthetic result", "structured_content": {"file_id": "file-1"}}


def test_remote_schema_is_complete_and_calls_use_trusted_identity():
    async def scenario():
        registry, client = ToolRegistry(), Client()
        tools = await register_dynamic_drive(registry, None, client=client, user_google_email="owner@example.com")
        tool = tools[0]
        assert tool.model_spec()["input_schema"]["properties"]["query"]["minLength"] == 3
        assert "user_google_email" not in tool.input_schema()["properties"]
        bad = await registry.execute_async(tool.name, {"query": "1"})
        assert not bad.success and not client.calls
        good = await registry.execute_async(tool.name, {"query": "abc"}, allowed_tool_names=[tool.name])
        assert good.success
        assert client.calls[0]["arguments"]["user_google_email"] == "owner@example.com"
        await register_dynamic_drive(registry, None, client=client, user_google_email="owner@example.com")
        assert len(registry.list_tools()) == 1
    asyncio.run(scenario())


def test_remote_schema_change_or_removal_rejects_execution():
    async def scenario():
        registry, client = ToolRegistry(), Client()
        tool = (await register_dynamic_drive(registry, None, client=client, user_google_email="owner@example.com"))[0]
        client.tools[0].input_schema["properties"]["query"]["minLength"] = 10
        assert not (await registry.execute_async(tool.name, {"query": "abc"})).success
        client.tools = []
        assert not (await registry.execute_async(tool.name, {"query": "abc"})).success
        assert not client.calls
    asyncio.run(scenario())


def test_unreviewed_tools_not_registered_and_upload_stays_controlled():
    async def scenario():
        registry, client = ToolRegistry(), Client()
        client.tools.extend([
            MCPToolInfo(name="mystery_read", description="claims read only", input_schema={"type": "object"}, annotations={"readOnlyHint": True}),
            MCPToolInfo(name="create_drive_file", description="upload", input_schema={"type": "object"}, annotations={"readOnlyHint": False}),
        ])
        tools = await register_dynamic_drive(registry, SimpleNamespace(), client=client, user_google_email="owner@example.com")
        assert len(tools) == 2
        upload = next(tool for tool in tools if "create_drive_file" in tool.name)
        assert "export_id" in upload.input_schema()["properties"]
        denied = await registry.execute_async(upload.name, {"export_id": "export-1"})
        assert not denied.success and not client.calls
        selected = available_tools_for_state(tools, observations={}, tool_attempt_counts={}, user_message="Search Drive files")
        assert upload not in selected
        assert selected
    asyncio.run(scenario())
