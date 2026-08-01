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
    RetryPolicy,
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
    return ChatCompletionsModelProvider(
        make_settings(),
        client=client,
        retry_policy=RetryPolicy(
            max_attempts=1,
            initial_delay_seconds=0,
            max_delay_seconds=0,
            total_timeout_seconds=10,
        ),
    )


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


def test_chat_completions_provider_retries_timeout_then_succeeds() -> None:
    attempts = 0
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.TimeoutException("temporary timeout")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "pong"}}]},
        )

    provider = ChatCompletionsModelProvider(
        make_settings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.5,
            max_delay_seconds=2,
            total_timeout_seconds=10,
        ),
        sleep=delays.append,
    )

    response = provider.generate(make_request())

    assert response.content == "pong"
    assert attempts == 2
    assert delays == [0.5]


def test_chat_completions_provider_retries_429_and_5xx() -> None:
    statuses = iter([429, 503, 200])
    delays = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if status == 200:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "recovered"}}]},
            )
        return httpx.Response(status, text="temporary provider failure")

    provider = ChatCompletionsModelProvider(
        make_settings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.5,
            max_delay_seconds=2,
            total_timeout_seconds=10,
        ),
        sleep=delays.append,
    )

    assert provider.generate(make_request()).content == "recovered"
    assert delays == [0.5, 1.0]


def test_chat_completions_provider_does_not_retry_nonrecoverable_4xx() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad request")

    provider = ChatCompletionsModelProvider(
        make_settings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_policy=RetryPolicy(max_attempts=3, total_timeout_seconds=10),
        sleep=lambda _: pytest.fail("400 response must not be retried"),
    )

    with pytest.raises(ModelHTTPError) as error:
        provider.generate(make_request())

    assert attempts == 1
    assert error.value.details["attempts"] == 1
    assert error.value.details["retry_exhausted"] is False


def test_chat_completions_provider_stops_when_total_budget_cannot_fit_delay() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="unavailable")

    provider = ChatCompletionsModelProvider(
        make_settings(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.5,
            max_delay_seconds=2,
            total_timeout_seconds=0.25,
        ),
        sleep=lambda _: pytest.fail("retry delay exceeds the total budget"),
    )

    with pytest.raises(ModelHTTPError) as error:
        provider.generate(make_request())

    assert attempts == 1
    assert error.value.details["retry_exhausted"] is True


def test_chat_completions_provider_rejects_missing_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ModelInvalidResponseError):
        make_provider(handler).generate(make_request())


def test_chat_completions_provider_rejects_invalid_structured_json() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
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

    with pytest.raises(ModelInvalidResponseError) as error:
        make_provider(handler).generate(make_request(RouteResult))

    # Direct generate() validates once. Production structured callers use
    # generate_structured(), which owns the one extra generation attempt.
    assert attempts == 1
    assert "structured_attempts" not in error.value.details


def test_generate_structured_retries_one_invalid_json_response() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        content = (
            "not-json"
            if attempts == 1
            else '{"intent": "policy_qa", "confidence": 0.8}'
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    response = make_provider(handler).generate_structured(
        messages=[ModelMessage(role=ModelRole.USER, content="Route this")],
        response_model=RouteResult,
    )

    assert attempts == 2
    assert response.structured == RouteResult(intent="policy_qa", confidence=0.8)


def test_generate_structured_stops_after_configured_attempts() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "still-not-json"}}]},
        )

    with pytest.raises(ModelInvalidResponseError) as error:
        make_provider(handler).generate_structured(
            messages=[ModelMessage(role=ModelRole.USER, content="Route this")],
            response_model=RouteResult,
        )

    assert attempts == 2
    assert error.value.details["structured_attempts"] == 2
