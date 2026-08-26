"""把已批准的 MCP 能力适配为普通 ToolDefinition。"""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Type

from pydantic import BaseModel

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolKind,
    ToolPermission,
    ToolResult,
)


class MCPClientProtocol(Protocol):
    async def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

    async def aclose(self) -> None:
        ...


@dataclass
class _MCPRequest:
    tool_name: str
    arguments: Dict[str, Any]
    future: asyncio.Future


class StdioMCPClient:
    """One long-lived stdio MCP process owned by one asyncio task.

    Keeping session creation, calls, and shutdown in the same task avoids AnyIO
    cancel-scope errors and prevents starting a Python MCP server for every call.
    """

    def __init__(
        self,
        *,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.command = command
        self.args = list(args)
        self.env = env
        self._queue: asyncio.Queue[Optional[_MCPRequest]] = asyncio.Queue()
        self._runner: Optional[asyncio.Task] = None
        self._ready: Optional[asyncio.Future] = None
        self._start_lock = asyncio.Lock()

    async def _ensure_started(self) -> None:
        async with self._start_lock:
            if self._runner is not None and self._runner.done():
                self._runner = None
                self._ready = None
            if self._runner is not None:
                assert self._ready is not None
                await self._ready
                return
            loop = asyncio.get_running_loop()
            self._ready = loop.create_future()
            self._runner = asyncio.create_task(self._run(), name="google-drive-mcp")
        assert self._ready is not None
        await self._ready

    async def _run(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        assert self._ready is not None
        try:
            parameters = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env,
            )
            async with AsyncExitStack() as stack:
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(parameters)
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                self._ready.set_result(None)
                while True:
                    request = await self._queue.get()
                    if request is None:
                        return
                    try:
                        result = await session.call_tool(
                            request.tool_name,
                            arguments=request.arguments,
                        )
                        if getattr(result, "isError", False):
                            request.future.set_exception(
                                RuntimeError(self._content_text(result.content))
                            )
                        else:
                            request.future.set_result(
                                {
                                    "text": self._content_text(result.content),
                                    "structured_content": getattr(
                                        result, "structuredContent", None
                                    ),
                                }
                            )
                    except Exception as error:
                        if not request.future.done():
                            request.future.set_exception(error)
                        return
        except Exception as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            while not self._queue.empty():
                request = self._queue.get_nowait()
                if request is not None and not request.future.done():
                    request.future.set_exception(error)

    @staticmethod
    def _content_text(content: Any) -> str:
        values = []
        for item in content or []:
            text = getattr(item, "text", None)
            if text:
                values.append(text)
        return "\n".join(values)

    async def call_tool(
        self,
        *,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        del server_name
        await self._ensure_started()
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(_MCPRequest(tool_name, arguments, future))
        return await future

    async def aclose(self) -> None:
        if self._runner is None:
            return
        await self._queue.put(None)
        await self._runner
        self._runner = None
        self._ready = None


def build_read_only_mcp_tool(
    *,
    name: str,
    description: str,
    server_name: str,
    remote_tool_name: str,
    input_model: Type[BaseModel],
    output_model: Type[BaseModel],
    client: MCPClientProtocol,
    domain: ToolDomain = ToolDomain.EXTERNAL,
    timeout_seconds: float = 20.0,
) -> ToolDefinition:
    """MCP 只改变传输方式，不绕过 Registry 权限和 Pydantic 校验。"""

    async def handler(arguments: BaseModel) -> ToolResult:
        data = await client.call_tool(
            server_name=server_name,
            tool_name=remote_tool_name,
            arguments=arguments.model_dump(mode="json"),
        )
        return ToolResult.ok(data=data, risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name=name,
        description=description,
        category=ToolCategory.SYSTEM,
        input_model=input_model,
        output_model=output_model,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        kind=ToolKind.MCP,
        domain=domain,
        parallel_safe=True,
        timeout_seconds=timeout_seconds,
        async_handler=handler,
    )
