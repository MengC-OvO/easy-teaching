from typing import Optional

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools import (
    ToolCategory,
    ToolDefinition,
    ToolErrorCode,
    ToolPermission,
    ToolResult,
    ToolTrace,
)


class ClassProfileInput(BaseModel):
    class_id: str


class ClassProfileOutput(BaseModel):
    class_id: str
    age_group: str


class CompactNestedInput(BaseModel):
    label: str = Field(description="A concise label.")


class CompactSchemaInput(BaseModel):
    query: str = Field(description="Search query.")
    nested: CompactNestedInput
    optional_limit: Optional[int] = 3


def fake_class_profile_handler(input_data: BaseModel) -> ToolResult:
    data = ClassProfileInput.model_validate(input_data)
    return ToolResult.ok(
        data={"class_id": data.class_id, "age_group": "3-5"},
        risk_level=RiskLevel.L0_READ_ONLY,
    )


def test_tool_definition_describes_a_registered_tool_contract() -> None:
    definition = ToolDefinition(
        name="get_class_context",
        description="Read a synthetic class profile by class id.",
        category=ToolCategory.CLASS_PROFILE,
        input_model=ClassProfileInput,
        output_model=ClassProfileOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=fake_class_profile_handler,
    )

    assert definition.name == "get_class_context"
    assert definition.category is ToolCategory.CLASS_PROFILE
    assert definition.risk_level is RiskLevel.L0_READ_ONLY
    assert definition.permission is ToolPermission.AUTO_EXECUTE
    assert definition.requires_approval is False
    assert definition.is_forbidden is False


def test_tool_definition_exposes_json_schemas_for_llm_tool_contracts() -> None:
    definition = ToolDefinition(
        name="get_class_context",
        description="Read a synthetic class profile by class id.",
        category=ToolCategory.CLASS_PROFILE,
        input_model=ClassProfileInput,
        output_model=ClassProfileOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=fake_class_profile_handler,
    )

    input_schema = definition.input_schema()
    output_schema = definition.output_schema()

    assert input_schema["properties"]["class_id"]["type"] == "string"
    assert "class_id" in input_schema["required"]
    assert output_schema["properties"]["age_group"]["type"] == "string"


def test_tool_definition_builds_compact_model_schema_without_validation_noise() -> None:
    definition = ToolDefinition(
        name="compact_test",
        description="Compact schema test.",
        category=ToolCategory.SYSTEM,
        input_model=CompactSchemaInput,
        output_model=ClassProfileOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=fake_class_profile_handler,
    )

    schema = definition.model_input_schema()

    assert schema["properties"]["query"] == {
        "type": "string",
        "description": "Search query.",
    }
    assert schema["properties"]["nested"]["properties"]["label"]["type"] == "string"
    assert schema["required"] == ["query", "nested"]
    assert "$defs" not in schema
    assert "title" not in str(schema)
    assert "default" not in str(schema)


def test_tool_definition_keeps_approval_metadata_with_tool() -> None:
    definition = ToolDefinition(
        name="save_observation",
        description="Save a draft after teacher approval.",
        category=ToolCategory.DRAFT,
        input_model=ClassProfileInput,
        output_model=ClassProfileOutput,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        permission=ToolPermission.REQUIRE_APPROVAL,
        handler=fake_class_profile_handler,
    )

    assert definition.requires_approval is True
    assert definition.is_forbidden is False


def test_tool_result_success_has_structured_metadata() -> None:
    result = ToolResult.ok(
        data={"class_id": "room-a"},
        risk_level=RiskLevel.L0_READ_ONLY,
        trace=ToolTrace(
            tool_name="get_class_context",
            risk_level=RiskLevel.L0_READ_ONLY,
            permission=ToolPermission.AUTO_EXECUTE,
            message="Read synthetic class profile.",
        ),
    )

    assert result.success is True
    assert result.data == {"class_id": "room-a"}
    assert result.error is None
    assert result.risk_level is RiskLevel.L0_READ_ONLY
    assert result.trace is not None
    assert result.trace.tool_name == "get_class_context"


def test_tool_result_failure_has_error_protocol() -> None:
    result = ToolResult.fail(
        code=ToolErrorCode.PERMISSION_DENIED,
        message="Tool requires teacher approval.",
        risk_level=RiskLevel.L2_CONTROLLED_WRITE,
        recoverable=True,
        details={"tool_name": "save_observation"},
    )

    assert result.success is False
    assert result.data == {}
    assert result.error is not None
    assert result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert result.error.recoverable is True
    assert result.error.details == {"tool_name": "save_observation"}
    assert result.risk_level is RiskLevel.L2_CONTROLLED_WRITE
