"""Live Google Drive MCP connectivity check without an LLM call."""

import argparse
import asyncio
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.asyncio_compat import run_async
from app.config import settings
from app.tools import StdioMCPClient


def _command() -> str:
    configured = settings.google_drive_mcp_command
    if configured != "workspace-mcp" or shutil.which(configured):
        return configured
    executable = "workspace-mcp.exe" if sys.platform == "win32" else "workspace-mcp"
    local = Path(sys.executable).with_name(executable)
    return str(local) if local.exists() else configured


async def _check(query: str, show_results: bool) -> None:
    if not settings.google_drive_mcp_enabled:
        raise SystemExit("GOOGLE_DRIVE_MCP_ENABLED must be true")
    if not all(
        (
            settings.google_drive_user_email,
            settings.google_oauth_client_id,
            settings.google_oauth_client_secret,
        )
    ):
        raise SystemExit("Google Drive MCP credentials are incomplete")

    credentials_dir = Path(settings.google_workspace_mcp_credentials_dir).resolve()
    client = StdioMCPClient(
        command=_command(),
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
            "WORKSPACE_MCP_CREDENTIALS_DIR": str(credentials_dir),
            "WORKSPACE_MCP_LOG_DIR": str(credentials_dir / "logs"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )
    try:
        arguments = {
            "user_google_email": settings.google_drive_user_email,
            "query": query,
            "page_size": 1,
            "detailed": False,
            "include_trashed": False,
        }
        try:
            result = await client.call_tool(
                server_name="google_workspace",
                tool_name="search_drive_files",
                arguments=arguments,
            )
        except RuntimeError as error:
            if "Authentication Needed" not in str(error):
                raise
            print("GOOGLE_DRIVE_AUTHORIZATION_REQUIRED")
            print("Complete the Google consent page opened in your browser.")
            credential_file = credentials_dir / (
                quote(settings.google_drive_user_email, safe="@._-") + ".json"
            )
            for _ in range(90):
                await asyncio.sleep(2)
                if credential_file.is_file():
                    break
            else:
                raise TimeoutError("Google authorization was not completed within 3 minutes")
            result = await client.call_tool(
                server_name="google_workspace",
                tool_name="search_drive_files",
                arguments=arguments,
            )
        text = result.get("text", "")
        print("GOOGLE_DRIVE_MCP_OK")
        print(f"response_received={bool(text)}")
        if show_results:
            print(text)
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="EasyTeaching")
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Print matching Drive file metadata; off by default for privacy.",
    )
    args = parser.parse_args()
    run_async(_check(args.query, args.show_results))


if __name__ == "__main__":
    main()
