import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.services import (
    ChatCompletionsModelProvider,
    ModelConfigurationError,
    ModelHTTPError,
    ModelInvalidResponseError,
    ModelMessage,
    ModelRequest,
    ModelRole,
    ModelTimeoutError,
)


class RouteResult(BaseModel):
    intent: str
    confidence: float


def make_settings() -> Settings:
    return Settings(
        MODEL_BASE_URL="https://model.test/v1",
        MODEL_CHAT_COMPLETIONS_PATH="/chat/completions",
        MODEL_API_KEY="test-key",
        MODEL_NAME="gemini-2.5-flash",
        MODEL_TIMEOUT_SECONDS=10,
    )


def make_request(response_model=None) -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="Return pong.")],
        response_model=response_model,
    )


def make_provider(handler) -> ChatCompletionsModelProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return ChatCompletionsModelProvider(make_settings(), client=client)


def test_chat_completions_provider_sends_openai_compatible_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://model.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = request.read()
        assert b"gemini-2.5-flash" in payload
        assert b"Return pong." in payload
        return httpx.Response(
            200,
            json={
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "message": {"content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            },
        )

    response = make_provider(handler).generate(make_request())

    assert response.content == "pong"
    assert response.model == "gemini-2.5-flash"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 6


def test_chat_completions_provider_parses_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "message": {
                            "content": '{"intent": "activity_planning", "confidence": 0.9}'
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    response = make_provider(handler).generate(make_request(RouteResult))

    assert response.structured == RouteResult(
        intent="activity_planning",
        confidence=0.9,
    )


def test_chat_completions_provider_parses_markdown_fenced_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "message": {
                            "content": (
                                "```json\n"
                                '{"intent": "activity_planning", "confidence": 0.9}\n'
                                "```"
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    response = make_provider(handler).generate(make_request(RouteResult))

    assert response.structured == RouteResult(
        intent="activity_planning",
        confidence=0.9,
    )


def test_chat_completions_provider_requires_configuration() -> None:
    provider = ChatCompletionsModelProvider(
        Settings(
            MODEL_BASE_URL="",
            MODEL_API_KEY="",
            MODEL_NAME="",
        ),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )

    with pytest.raises(ModelConfigurationError) as error:
        provider.generate(make_request())

    assert error.value.details["missing"] == [
        "MODEL_BASE_URL",
        "MODEL_API_KEY",
        "MODEL_NAME",
    ]


def test_chat_completions_provider_wraps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    with pytest.raises(ModelTimeoutError):
        make_provider(handler).generate(make_request())


def test_chat_completions_provider_wraps_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    with pytest.raises(ModelHTTPError) as error:
        make_provider(handler).generate(make_request())

    assert error.value.recoverable is True
    assert error.value.details["status_code"] == 429


def test_chat_completions_provider_rejects_missing_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ModelInvalidResponseError):
        make_provider(handler).generate(make_request())


def test_chat_completions_provider_rejects_invalid_structured_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "not-json"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    with pytest.raises(ModelInvalidResponseError):
        make_provider(handler).generate(make_request(RouteResult))
