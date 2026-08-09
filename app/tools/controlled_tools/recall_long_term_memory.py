"""A scoped, read-only memory lookup for ReAct workflows."""

import inspect
from typing import Any, List

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolErrorCode,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


class RecallLongTermMemoryInput(BaseModel):
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=5)


class RecalledLongTermMemory(BaseModel):
    memory_id: str
    memory_type: str
    content: str
    importance: int


class RecallLongTermMemoryOutput(BaseModel):
    memories: List[RecalledLongTermMemory] = Field(default_factory=list)


def build_recall_long_term_memory_tool(store: Any) -> ToolDefinition:
    def runtime_handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        data = RecallLongTermMemoryInput.model_validate(input_data)
        if context.teacher_id is None and context.class_id is None:
            return ToolResult.fail(
                code=ToolErrorCode.VALIDATION_ERROR,
                message="Long-term memory recall requires a teacher_id or class_id.",
                risk_level=RiskLevel.L0_READ_ONLY,
                recoverable=True,
            )
        memories = store.search_recall_memories(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            query=data.query,
            limit=data.limit,
        )
        return ToolResult.ok(
            data={
                "memories": [
                    {
                        "memory_id": memory["memory_id"],
                        "memory_type": memory["memory_type"],
                        "content": memory["content"],
                        "importance": int(memory["importance"]),
                    }
                    for memory in memories
                ]
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    async def async_runtime_handler(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        data = RecallLongTermMemoryInput.model_validate(input_data)
        if context.teacher_id is None and context.class_id is None:
            return ToolResult.fail(
                code=ToolErrorCode.VALIDATION_ERROR,
                message="Long-term memory recall requires a teacher_id or class_id.",
                risk_level=RiskLevel.L0_READ_ONLY,
            )
        memories = store.search_recall_memories(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            query=data.query,
            limit=data.limit,
        )
        if inspect.isawaitable(memories):
            memories = await memories
        return ToolResult.ok(
            data={
                "memories": [
                    {
                        "memory_id": item["memory_id"],
                        "memory_type": item["memory_type"],
                        "content": item["content"],
                        "importance": int(item["importance"]),
                    }
                    for item in memories
                ]
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="recall_long_term_memory",
        description=(
            "Search task-specific durable memories for the current teacher or "
            "class. Use only when historical detail would improve the draft."
        ),
        category=ToolCategory.MEMORY,
        input_model=RecallLongTermMemoryInput,
        output_model=RecallLongTermMemoryOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        runtime_handler=runtime_handler,
        async_runtime_handler=async_runtime_handler,
    )
