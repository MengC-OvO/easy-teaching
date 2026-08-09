import asyncio

from pydantic import BaseModel

from app.schemas import RiskLevel
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolErrorCode,
    ToolPermission,
    ToolRegistry,
    ToolResult,
)


class ValueInput(BaseModel):
    value: str


class ValueOutput(BaseModel):
    value: str


def build_registry(*, delay: float = 0.0, timeout: float = 1.0) -> ToolRegistry:
    async def handler(input_data: BaseModel) -> ToolResult:
        data = ValueInput.model_validate(input_data)
        if delay:
            await asyncio.sleep(delay)
        return ToolResult.ok(
            data={"value": data.value},
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="value",
            description="Return one value.",
            category=ToolCategory.SYSTEM,
            input_model=ValueInput,
            output_model=ValueOutput,
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            domain=ToolDomain.INTERNAL,
            parallel_safe=True,
            timeout_seconds=timeout,
            async_handler=handler,
        )
    )
    return registry


def test_execute_async_reuses_validation_and_execution() -> None:
    result = asyncio.run(
        build_registry().execute_async("value", {"value": "hello"})
    )

    assert result.success is True
    assert result.data == {"value": "hello"}


def test_execute_async_timeout_returns_recoverable_error() -> None:
    result = asyncio.run(
        build_registry(delay=0.05, timeout=0.001).execute_async(
            "value",
            {"value": "slow"},
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code is ToolErrorCode.TIMEOUT
    assert result.error.recoverable is True
