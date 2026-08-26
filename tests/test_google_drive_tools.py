import hashlib
import asyncio
from pathlib import Path

from app.tools import ToolExecutionContext, ToolPermission, build_google_drive_tools


class StubMCPClient:
    def __init__(self):
        self.calls = []

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


def test_drive_search_injects_configured_email_and_is_read_only(tmp_path):
    client = StubMCPClient()
    tools = build_google_drive_tools(
        StubExportStore(tmp_path / "unused"),
        client=client,
        user_google_email="teacher@example.com",
        export_root=tmp_path,
    )
    search = tools[0]

    result = asyncio.run(
        search.async_handler(search.input_model(query="program plan", limit=4))
    )

    assert result.success
    assert search.permission is ToolPermission.AUTO_EXECUTE
    assert client.calls[0]["tool_name"] == "search_drive_files"
    assert client.calls[0]["arguments"]["user_google_email"] == "teacher@example.com"
    assert client.calls[0]["arguments"]["page_size"] == 4


def test_drive_upload_accepts_only_managed_export_and_requires_approval(tmp_path):
    export = tmp_path / "record.pdf"
    export.write_bytes(b"safe export")
    client = StubMCPClient()
    tools = build_google_drive_tools(
        StubExportStore(export),
        client=client,
        user_google_email="teacher@example.com",
        export_root=tmp_path,
    )
    upload = tools[1]

    result = asyncio.run(
        upload.async_runtime_handler(
            upload.input_model(export_id="export-1", folder_id="folder-1"),
            ToolExecutionContext(teacher_id="teacher-1", class_id="class-1"),
        )
    )

    assert result.success
    assert upload.permission is ToolPermission.REQUIRE_APPROVAL
    assert client.calls[0]["tool_name"] == "create_drive_file"
    arguments = client.calls[0]["arguments"]
    assert arguments["folder_id"] == "folder-1"
    assert arguments["base64_content"] == "c2FmZSBleHBvcnQ="
    assert "file_path" not in arguments


def test_drive_upload_rejects_path_outside_managed_export_root(tmp_path):
    export_root = tmp_path / "exports"
    export_root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    client = StubMCPClient()
    upload = build_google_drive_tools(
        StubExportStore(outside),
        client=client,
        user_google_email="teacher@example.com",
        export_root=export_root,
    )[1]

    result = asyncio.run(
        upload.async_runtime_handler(
            upload.input_model(export_id="export-1"),
            ToolExecutionContext(teacher_id="teacher-1", class_id="class-1"),
        )
    )

    assert not result.success
    assert not client.calls
