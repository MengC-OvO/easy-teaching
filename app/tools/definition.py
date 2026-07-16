from enum import Enum
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, Field

from app.schemas import RiskLevel


class ToolPermission(str, Enum):
    AUTO_EXECUTE = "auto_execute"
    REQUIRE_APPROVAL = "require_approval"
    FORBIDDEN = "forbidden"


class ToolCategory(str, Enum):
    CLASS_PROFILE = "class_profile"
    DRAFT = "draft"
    POLICY = "policy"
    SYSTEM = "system"


class ToolErrorCode(str, Enum):
    VALIDATION_ERROR = "validation_error"
    TOOL_NOT_FOUND = "tool_not_found"
    PERMISSION_DENIED = "permission_denied"
    EXECUTION_ERROR = "execution_error"


class ToolError(BaseModel):
    code: ToolErrorCode
    message: str
    recoverable: bool = True
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolTrace(BaseModel):
    tool_name: str
    risk_level: RiskLevel
    permission: ToolPermission
    message: str


class ToolResult(BaseModel):
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[ToolError] = None
    risk_level: RiskLevel
    trace: Optional[ToolTrace] = None

    @classmethod
    def ok(
        cls,
        *,
        data: Optional[Dict[str, Any]] = None,
        risk_level: RiskLevel,
        trace: Optional[ToolTrace] = None,
    ) -> "ToolResult":
        return cls(
            success=True,
            data=data or {},
            risk_level=risk_level,
            trace=trace,
        )

    @classmethod
    def fail(
        cls,
        *,
        code: ToolErrorCode,
        message: str,
        risk_level: RiskLevel,
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
        trace: Optional[ToolTrace] = None,
    ) -> "ToolResult":
        return cls(
            success=False,
            error=ToolError(
                code=code,
                message=message,
                recoverable=recoverable,
                details=details or {},
            ),
            risk_level=risk_level,
            trace=trace,
        )


ToolHandler = Callable[[BaseModel], ToolResult]


class ToolDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: ToolCategory
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    risk_level: RiskLevel
    permission: ToolPermission
    handler: ToolHandler

    model_config = {"arbitrary_types_allowed": True}

    @property
    def requires_approval(self) -> bool:
        return self.permission is ToolPermission.REQUIRE_APPROVAL

    @property
    def is_forbidden(self) -> bool:
        return self.permission is ToolPermission.FORBIDDEN

    def input_schema(self) -> Dict[str, Any]:
        return self.input_model.model_json_schema()

    def output_schema(self) -> Dict[str, Any]:
        return self.output_model.model_json_schema()
