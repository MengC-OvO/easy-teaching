from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.graph_state import Intent


class IntentRouteResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_clarification_question(self) -> "IntentRouteResult":
        if self.needs_clarification and not self.clarification_question:
            raise ValueError("clarification_question is required when needs_clarification is true")
        if not self.needs_clarification and self.clarification_question:
            raise ValueError(
                "clarification_question should be empty when needs_clarification is false"
            )
        return self

    @property
    def is_confident_route(self) -> bool:
        return not self.needs_clarification and self.intent is not Intent.UNKNOWN
