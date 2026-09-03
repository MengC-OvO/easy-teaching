import hashlib
import asyncio
from pathlib import Path

from app.tools import (
    MCPToolInfo,
    ToolExecutionContext,
    ToolErrorCode,
    ToolPermission,
    ToolRegistry,
    build_google_drive_tool,
)


class StubMCPClient:
    def __init__(self):
        self.calls = []
        self.list_calls = 0

    async def list_tools(self, **values):
        self.list_calls += 1
        return [
            MCPToolInfo(
                name="search_drive_files",
                description="Search Drive",
                input_schema={
                    "type": "object",
                    "properties": {
                        "user_google_email": {"type": "string"},
                        "query": {"type": "string"},
                        "page_size": {"type": "integer"},
                    },
                    "required": ["user_google_email", "query"],
                },
                annotations={"readOnlyHint": True, "destructiveHint": False},
            ),
            MCPToolInfo(
                name="create_drive_file",
                description="Create a Drive file",
                input_schema={"type": "object"},
                annotations={"readOnlyHint": False, "destructiveHint": False},
            ),
        ]

    async def call_tool(self, **values):
        self.calls.append(values)
        return {"text": "Drive operation completed."}

    async def aclose(self):
        return None


class StubExportStore:
    def __init__(self, path: Path):
        self.path = path

    async def get_record_export(self, **values):
        return {
            "storage_path": str(self.path),
            "checksum": hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }


def test_drive_gateway_discovers_then_executes_read_only_tool(tmp_path):
    client = StubMCPClient()
    tool = build_google_drive_tool(
        StubExportStore(tmp_path / "unused"),
        client=client,
        user_google_email="teacher@example.com",
        export_root=tmp_path,
    )
    registry = ToolRegistry()
    registry.register(tool)

    assert client.list_calls == 0

    discovered = asyncio.run(
        registry.execute_async(
            "drive_operation",
            {"action": "discover", "intent": "search for a program plan"},
        )
    )
    assert discovered.success
    assert client.list_calls == 1
    search_spec = next(
        item for item in discovered.data["tools"] if item["name"] == "search_drive_files"
    )
    assert "user_google_email" not in search_spec["input_schema"]["properties"]
    assert tool.permission_for(
        {
            "action": "execute",
            "tool_name": "search_drive_files",
            "arguments": {"query": "program plan", "page_size": 4},
        }
    ) is ToolPermission.AUTO_EXECUTE

    result = asyncio.run(
        registry.execute_async(
            "drive_operation",
            {
                "action": "execute",
                "tool_name": "search_drive_files",
                "arguments": {"query": "program plan", "page_size": 4},
            },
        )
    )

    assert result.success
    assert client.list_calls == 1
    assert client.calls[0]["tool_name"] == "search_drive_files"
    assert client.calls[0]["arguments"]["user_google_email"] == "teacher@example.com"
    assert client.calls[0]["arguments"]["page_size"] == 4


def test_drive_upload_accepts_only_managed_export_and_requires_approval(tmp_path):
    export = tmp_path / "record.pdf"
    export.write_bytes(b"safe export")
    client = StubMCPClient()
    tool = build_google_drive_tool(
        StubExportStore(export),
        client=client,
        user_google_email="teacher@example.com",
        export_root=tmp_path,
    )
    registry = ToolRegistry()
    registry.register(tool)
    asyncio.run(
        registry.execute_async(
            "drive_operation",
            {"action": "discover", "intent": "upload an export"},
        )
    )
    raw_call = {
        "action": "execute",
        "tool_name": "create_drive_file",
        "arguments": {"export_id": "export-1", "folder_id": "folder-1"},
    }

    blocked = asyncio.run(
        registry.execute_async(
            "drive_operation",
            raw_call,
            execution_context=ToolExecutionContext(
                teacher_id="teacher-1", class_id="class-1"
            ),
        )
    )
    assert not blocked.success
    assert not client.calls

    result = asyncio.run(
        registry.execute_async(
            "drive_operation",
            raw_call,
            approved=True,
            execution_context=ToolExecutionContext(
                teacher_id="teacher-1", class_id="class-1"
            ),
        )
    )

    assert result.success
    assert tool.permission_for(raw_call) is ToolPermission.REQUIRE_APPROVAL
    assert client.calls[0]["tool_name"] == "create_drive_file"
    arguments = client.calls[0]["arguments"]
    assert arguments["folder_id"] == "folder-1"
    assert arguments["base64_content"] == "c2FmZSBleHBvcnQ="
    assert "file_path" not in arguments


def test_drive_gateway_refuses_execute_before_discovery(tmp_path):
    client = StubMCPClient()
    tool = build_google_drive_tool(
        StubExportStore(tmp_path / "unused"),
        client=client,
        user_google_email="teacher@example.com",
        export_root=tmp_path,
    )
    registry = ToolRegistry()
    registry.register(tool)

    result = asyncio.run(
        registry.execute_async(
            "drive_operation",
            {
                "action": "execute",
                "tool_name": "search_drive_files",
                "arguments": {"query": "program plan"},
            },
            approved=True,
        )
    )

    assert not result.success
    assert result.error.code is ToolErrorCode.TOOL_NOT_FOUND
    assert client.list_calls == 0
    assert not client.calls


def test_drive_gateway_rejects_remote_file_url_payload(tmp_path):
    client = StubMCPClient()

    async def list_import_tool(**values):
        client.list_calls += 1
        return [
            MCPToolInfo(
                name="import_to_google_doc",
                description="Import a document",
                input_schema={
                    "type": "object",
                    "properties": {
                        "user_google_email": {"type": "string"},
                        "file_name": {"type": "string"},
                        "file_url": {"type": "string"},
                    },
                    "required": ["user_google_email", "file_name"],
                },
                annotations={"readOnlyHint": False, "destructiveHint": False},
            )
        ]

    client.list_tools = list_import_tool
    tool = build_google_drive_tool(
        StubExportStore(tmp_path / "unused"),
        client=client,
        user_google_email="teacher@example.com",
        export_root=tmp_path,
    )
    registry = ToolRegistry()
    registry.register(tool)
    asyncio.run(
        registry.execute_async(
            "drive_operation",
            {"action": "discover", "intent": "import a document"},
        )
    )

    result = asyncio.run(
        registry.execute_async(
            "drive_operation",
            {
                "action": "execute",
                "tool_name": "import_to_google_doc",
                "arguments": {
                    "file_name": "document.docx",
                    "file_url": "file:///private/document.docx",
                },
            },
            approved=True,
        )
    )

    assert not result.success
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert client.list_calls == 1
    assert not client.calls


def test_drive_upload_rejects_path_outside_managed_export_root(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    client = StubMCPClient()
    tool = build_google_drive_tool(
        StubExportStore(outside),
        client=client,
        user_google_email="teacher@example.com",
        export_root=export_root,
    )
    registry = ToolRegistry()
    registry.register(tool)
    asyncio.run(
        registry.execute_async(
            "drive_operation",
            {"action": "discover", "intent": "upload an export"},
        )
    )

    result = asyncio.run(
        registry.execute_async(
            "drive_operation",
            {
                "action": "execute",
                "tool_name": "create_drive_file",
                "arguments": {"export_id": "export-1"},
            },
            approved=True,
            execution_context=ToolExecutionContext(
                teacher_id="teacher-1", class_id="class-1"
            ),
        )
    )

    assert not result.success
    assert not client.calls


def test_drive_gateway_requires_approval_if_search_is_marked_as_write(tmp_path):
    client = StubMCPClient()

    async def list_write_tool(**values):
        return [
            MCPToolInfo(
                name="search_drive_files",
                description="Suspicious search",
                input_schema={"type": "object"},
                annotations={"readOnlyHint": False, "destructiveHint": False},
            )
        ]

    client.list_tools = list_write_tool
    tool = build_google_drive_tool(
        StubExportStore(tmp_path / "unused"),
        client=client,
        user_google_email="teacher@example.com",
        export_root=tmp_path,
    )

    registry = ToolRegistry()
    registry.register(tool)
    asyncio.run(
        registry.execute_async(
            "drive_operation",
            {"action": "discover", "intent": "search"},
        )
    )
    call = {
        "action": "execute",
        "tool_name": "search_drive_files",
        "arguments": {},
    }

    assert tool.permission_for(call) is ToolPermission.REQUIRE_APPROVAL
    assert not client.calls
