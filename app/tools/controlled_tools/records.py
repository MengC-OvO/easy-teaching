import inspect
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas import RiskLevel
from app.tools.definition import (
    PreparedToolAction,
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolErrorCode,
    ToolExecutionContext,
    ToolPermission,
    ToolResult,
)


class QueryRecordsInput(BaseModel):
    record_type: Literal["observation", "educational_record", "all"] = "all"
    search_text: Optional[str] = Field(default=None, min_length=2, max_length=300)
    child_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    status: Optional[Literal["draft", "final", "archived"]] = None
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def validate_dates(self) -> "QueryRecordsInput":
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        return self


class QueryRecordsOutput(BaseModel):
    records: List[Dict[str, Any]] = Field(default_factory=list)
    returned_count: int = Field(ge=0)
    search_text: Optional[str] = None


class SaveObservationInput(BaseModel):
    child_ids: List[str] = Field(default_factory=list, max_length=20)
    observed_at: datetime
    setting: str = Field(min_length=2, max_length=300)
    objective_text: Optional[str] = Field(default=None, min_length=10, max_length=10_000)
    educator_actions: Optional[str] = Field(default=None, max_length=5_000)
    status: Literal["draft", "final"] = "draft"
    source_request_id: Optional[str] = Field(default=None, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def require_text_or_source(self) -> "SaveObservationInput":
        if not self.objective_text and not self.source_request_id:
            raise ValueError("objective_text or source_request_id is required")
        return self


class SaveObservationOutput(BaseModel):
    record_type: Literal["observation"]
    observation_id: str
    centre_id: str
    class_id: str
    author_teacher_id: str
    child_ids: List[str]
    observed_at: str
    setting: str
    objective_text: str
    educator_actions: Optional[str] = None
    status: str
    version: int
    created_at: str
    updated_at: str


class CurriculumLink(BaseModel):
    framework: str = Field(min_length=2, max_length=80)
    outcome: str = Field(min_length=1, max_length=300)
    citation_id: Optional[str] = Field(default=None, max_length=128)


class SaveEducationalRecordInput(BaseModel):
    record_type: Literal[
        "learning_story", "learning_analysis", "program_plan", "reflection", "follow_up"
    ]
    title: Optional[str] = Field(default=None, min_length=2, max_length=300)
    analysis: Optional[str] = Field(default=None, min_length=20, max_length=20_000)
    curriculum_links: List[CurriculumLink] = Field(default_factory=list, max_length=20)
    next_steps: List[str] = Field(default_factory=list, max_length=20)
    observation_ids: List[str] = Field(default_factory=list, max_length=50)
    status: Literal["draft", "final"] = "draft"
    source_request_id: Optional[str] = Field(default=None, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def require_content_or_source(self) -> "SaveEducationalRecordInput":
        if not self.source_request_id and (not self.title or not self.analysis):
            raise ValueError("title and analysis are required without source_request_id")
        return self


class SaveEducationalRecordOutput(BaseModel):
    record_type: Literal["educational_record"]
    educational_record_type: str
    record_id: str
    centre_id: str
    class_id: str
    author_teacher_id: str
    title: str
    analysis: str
    curriculum_links: List[Dict[str, Any]]
    next_steps: List[str]
    observation_ids: List[str]
    status: str
    version: int
    created_at: str
    updated_at: str


def _trusted_scope(context: ToolExecutionContext) -> Optional[ToolResult]:
    if context.teacher_id and context.class_id:
        return None
    return ToolResult.fail(
        code=ToolErrorCode.PERMISSION_DENIED,
        message="Record access requires a trusted teacher and class scope.",
        risk_level=RiskLevel.L3_FORBIDDEN,
        recoverable=False,
    )


def build_query_records_tool(store: Any) -> ToolDefinition:
    async def async_runtime_handler(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = _trusted_scope(context)
        if denied:
            return denied
        data = QueryRecordsInput.model_validate(input_data)
        result = store.query_records(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            **data.model_dump(),
        )
        if inspect.isawaitable(result):
            result = await result
        output = QueryRecordsOutput(
            records=result,
            returned_count=len(result),
            search_text=data.search_text,
        )
        return ToolResult.ok(data=output.model_dump(mode="json"), risk_level=RiskLevel.L0_READ_ONLY)

    return ToolDefinition(
        name="query_records",
        description=(
            "Query authorised observations or educational records. Call this only "
            "when the teacher asks to use prior records or the current task explicitly "
            "depends on them; do not call it for a simple new observation. One "
            "completed exact title/ID lookup is conclusive even when it returns zero."
        ),
        category=ToolCategory.DRAFT,
        input_model=QueryRecordsInput,
        output_model=QueryRecordsOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.LOCAL,
        parallel_safe=True,
        async_runtime_handler=async_runtime_handler,
    )


def build_save_observation_tool(store: Any) -> ToolDefinition:
    async def prepare_action(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> PreparedToolAction:
        data = SaveObservationInput.model_validate(input_data)
        payload = data.model_dump(mode="json")
        if not data.objective_text:
            source = await _source_draft(store, data.source_request_id, context)
            payload["objective_text"] = source["content"]
        return PreparedToolAction(arguments=payload, preview=payload)

    async def async_runtime_handler(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = _trusted_scope(context)
        if denied:
            return denied
        data = SaveObservationInput.model_validate(input_data)
        result = store.save_observation(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            **data.model_dump(),
        )
        if inspect.isawaitable(result):
            result = await result
        return ToolResult.ok(data=result, risk_level=RiskLevel.L2_CONTROLLED_WRITE)

    return ToolDefinition(
        name="save_observation",
        description=(
            "Save a reviewed objective observation after approval. For a conversation "
            "draft, pass source_request_id and omit objective_text; the server resolves "
            "and freezes the full draft. Do not retrieve EYLF or prior records unless asked."
        ),
        category=ToolCategory.DRAFT,
        input_model=SaveObservationInput,
        output_model=SaveObservationOutput,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        completion_aliases=(
            "保存为观察记录",
            "保存观察记录",
            "save as an observation",
            "save the observation",
        ),
        domain=ToolDomain.LOCAL,
        parallel_safe=False,
        async_runtime_handler=async_runtime_handler,
        approval_preparation_handler=prepare_action,
    )


def build_save_educational_record_tool(store: Any) -> ToolDefinition:
    async def prepare_action(
        input_data: BaseModel,
        context: ToolExecutionContext,
    ) -> PreparedToolAction:
        data = SaveEducationalRecordInput.model_validate(input_data)
        payload = data.model_dump(mode="json")
        if not data.title or not data.analysis:
            source = await _source_draft(store, data.source_request_id, context)
            payload["title"] = data.title or source["title"] or "Educational record"
            payload["analysis"] = data.analysis or source["content"]
        return PreparedToolAction(arguments=payload, preview=payload)

    async def async_runtime_handler(input_data: BaseModel, context: ToolExecutionContext) -> ToolResult:
        denied = _trusted_scope(context)
        if denied:
            return denied
        data = SaveEducationalRecordInput.model_validate(input_data)
        payload = data.model_dump()
        payload["curriculum_links"] = [item.model_dump() for item in data.curriculum_links]
        result = store.save_educational_record(
            teacher_id=context.teacher_id,
            class_id=context.class_id,
            **payload,
        )
        if inspect.isawaitable(result):
            result = await result
        return ToolResult.ok(data=result, risk_level=RiskLevel.L2_CONTROLLED_WRITE)

    return ToolDefinition(
        name="save_educational_record",
        description=(
            "Save an approved learning story, analysis, program plan, reflection or "
            "follow-up. For a conversation draft, pass source_request_id and omit "
            "title/analysis; the server resolves and freezes the full content."
        ),
        category=ToolCategory.DRAFT,
        input_model=SaveEducationalRecordInput,
        output_model=SaveEducationalRecordOutput,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        completion_aliases=(
            "保存为教育记录",
            "保存教育记录",
            "保存为学习故事",
            "save as an educational record",
            "save the educational record",
            "save as a learning story",
        ),
        domain=ToolDomain.LOCAL,
        parallel_safe=False,
        async_runtime_handler=async_runtime_handler,
        approval_preparation_handler=prepare_action,
    )


async def _source_draft(
    store: Any,
    source_request_id: Optional[str],
    context: ToolExecutionContext,
) -> Dict[str, str]:
    if not source_request_id or not context.session_id:
        raise ValueError("A source draft requires trusted request and session references")
    source = store.get_conversation_run_result(source_request_id)
    if inspect.isawaitable(source):
        source = await source
    if source is None or source.get("session_id") != context.session_id:
        raise ValueError("The source draft is unavailable in this conversation")
    draft = source.get("draft") or {}
    content = str(draft.get("content") or "").strip()
    if not content:
        raise ValueError("The source draft has no content")
    return {
        "title": str(draft.get("title") or "").strip(),
        "content": content,
    }
