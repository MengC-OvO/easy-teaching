import sys
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import (
    ChatCompletionsModelProvider,
    ModelMessage,
    ModelProviderError,
    ModelRequest,
    ModelRole,
)


class SmokeRouteResult(BaseModel):
    intent: str = Field(pattern="^(activity_planning|learning_record|policy_qa|family_communication)$")
    confidence: float = Field(ge=0.0, le=1.0)


def run_text_smoke_test(provider: ChatCompletionsModelProvider) -> None:
    response = provider.generate(
        ModelRequest(
            messages=[
                ModelMessage(
                    role=ModelRole.USER,
                    content="Reply with exactly one word: pong",
                )
            ],
            temperature=0.0,
        )
    )

    print("TEXT_SMOKE_OK")
    print(f"model={response.model}")
    print(f"finish_reason={response.finish_reason}")
    print(f"content={response.content!r}")


def run_structured_smoke_test(provider: ChatCompletionsModelProvider) -> None:
    response = provider.generate_structured(
        messages=[
            ModelMessage(
                role=ModelRole.SYSTEM,
                content=(
                    "Return only valid JSON. No markdown. "
                    "The JSON must match: {\"intent\": string, \"confidence\": number}."
                ),
            ),
            ModelMessage(
                role=ModelRole.USER,
                content=(
                    "Classify this teacher request: "
                    "Please plan an outdoor sensory activity for preschool children. "
                    "Use intent activity_planning and confidence 0.9."
                ),
            ),
        ],
        response_model=SmokeRouteResult,
        temperature=0.0,
    )

    print("STRUCTURED_SMOKE_OK")
    print(f"structured={response.structured.model_dump()}")


def main() -> int:
    provider = ChatCompletionsModelProvider()
    try:
        run_text_smoke_test(provider)
        run_structured_smoke_test(provider)
    except ModelProviderError as error:
        print("MODEL_SMOKE_FAILED")
        print(error.to_dict())
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
