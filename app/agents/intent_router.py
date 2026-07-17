from typing import List, Optional, Protocol, Type

from app.schemas import IntentRouteResult
from app.services import (
    ChatCompletionsModelProvider,
    ModelMessage,
    ModelResponse,
    ModelRole,
)


INTENT_ROUTER_SYSTEM_PROMPT = """
You are the intent router for EduFlow AU, a teacher workflow assistant for
Australian early childhood education scenarios.

Classify the teacher request into exactly one of these intents:
- activity_planning: activity ideas, lesson/activity plans, materials, steps, observation points
- learning_record: learning stories, observation notes, reflections, documentation drafts
- policy_qa: policy, EYLF, NQS, centre policy, compliance-oriented questions
- family_communication: messages or updates drafted for families/parents/carers
- unknown: unclear, unsupported, or ambiguous request

Return only valid JSON matching this schema:
{
  "intent": "activity_planning | learning_record | policy_qa | family_communication | unknown",
  "confidence": number between 0 and 1,
  "needs_clarification": boolean,
  "clarification_question": string or null,
  "reason": string
}

Rules:
- Do not answer the user request.
- Do not generate plans, policy answers, or messages.
- Do not call tools.
- Use needs_clarification=true when the request is ambiguous or confidence is below 0.65.
- If needs_clarification=true, intent must be unknown and clarification_question must be a direct question.
- If needs_clarification=false, clarification_question must be null.
""".strip()


class StructuredModelProvider(Protocol):
    def generate_structured(
        self,
        *,
        messages: List[ModelMessage],
        response_model: Type[IntentRouteResult],
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...


class IntentRouter:
    def __init__(self, provider: Optional[StructuredModelProvider] = None) -> None:
        self.provider = provider or ChatCompletionsModelProvider()

    def route(self, user_message: str) -> IntentRouteResult:
        response = self.provider.generate_structured(
            messages=[
                ModelMessage(role=ModelRole.SYSTEM, content=INTENT_ROUTER_SYSTEM_PROMPT),
                ModelMessage(role=ModelRole.USER, content=user_message),
            ],
            response_model=IntentRouteResult,
            temperature=0.0,
        )

        if not isinstance(response.structured, IntentRouteResult):
            raise TypeError("Intent router provider returned an unexpected structured result")
        return response.structured
