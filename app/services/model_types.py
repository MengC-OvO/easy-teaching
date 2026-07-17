from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field


class ModelRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ModelMessage(BaseModel):
    role: ModelRole
    content: str = Field(min_length=1)


class ModelUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ModelRequest(BaseModel):
    messages: List[ModelMessage] = Field(min_length=1)
    model: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout_seconds: Optional[float] = Field(default=None, gt=0)
    response_model: Optional[Type[BaseModel]] = None

    model_config = {"arbitrary_types_allowed": True}


class ModelResponse(BaseModel):
    content: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    structured: Optional[BaseModel] = None

    model_config = {"arbitrary_types_allowed": True}


class ModelJSONParseResult(BaseModel):
    data: Dict[str, Any]
