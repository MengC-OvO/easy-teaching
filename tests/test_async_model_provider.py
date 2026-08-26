import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.services import (
    ChatCompletionsModelProvider,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelRole,
    RetryPolicy,
)


class StructuredAnswer(BaseModel):
    answer: str


def _provider(handler, *, attempts=1):
    return ChatCompletionsModelProvider(
        Settings(
            MODEL_BASE_URL="https://model.test/v1",
            MODEL_API_KEY="test-key",
            MODEL_NAME="test-model",
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry_policy=RetryPolicy(
            max_attempts=attempts,
            initial_delay_seconds=0,
            max_delay_seconds=0,
            total_timeout_seconds=10,
        ),
    )


def test_async_provider_generates_and_parses_structured_output():
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        schema_instruction = payload["messages"][-1]["content"]
        assert '"answer"' in schema_instruction
        assert "exact field names" in schema_instruction
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer":"ok"}'}}]},
        )

    async def run():
        provider = _provider(handler)
        try:
            return await provider.generate_structured(
                messages=[ModelMessage(role=ModelRole.USER, content="Answer")],
                response_model=StructuredAnswer,
            )
        finally:
            await provider.client.aclose()

    response = asyncio.run(run())
    assert response.structured == StructuredAnswer(answer="ok")


def test_async_provider_adds_validation_feedback_before_structured_retry():
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        content = '{"wrong":"value"}' if len(payloads) == 1 else '{"answer":"ok"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def run():
        provider = _provider(handler)
        try:
            return await provider.generate_structured(
                messages=[ModelMessage(role=ModelRole.USER, content="Answer")],
                response_model=StructuredAnswer,
            )
        finally:
            await provider.client.aclose()

    response = asyncio.run(run())
    assert response.structured == StructuredAnswer(answer="ok")
    assert len(payloads) == 2
    assert "previous structured response was invalid" in payloads[1]["messages"][-2][
        "content"
    ].lower()
    assert "answer" in payloads[1]["messages"][-2]["content"]


def test_async_provider_retries_recoverable_http_error():
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503, text="temporary")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def run():
        provider = _provider(handler, attempts=2)
        try:
            return await provider.generate(
                ModelRequest(messages=[ModelMessage(role=ModelRole.USER, content="Answer")])
            )
        finally:
            await provider.client.aclose()

    assert asyncio.run(run()).content == "ok"
    assert len(seen) == 2


def test_async_provider_does_not_retry_nonrecoverable_error():
    provider = _provider(lambda request: httpx.Response(400, text="bad"), attempts=3)

    async def run():
        try:
            with pytest.raises(ModelHTTPError):
                await provider.generate(
                    ModelRequest(messages=[ModelMessage(role=ModelRole.USER, content="Answer")])
                )
        finally:
            await provider.client.aclose()

    asyncio.run(run())


def test_async_provider_accepts_one_json_object_wrapped_in_provider_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": 'Here is the result:\n{"answer":"ok"}\nDone.'
                        }
                    }
                ]
            },
        )

    async def run():
        provider = _provider(handler)
        try:
            response = await provider.generate_structured(
                messages=[ModelMessage(role=ModelRole.USER, content="test")],
                response_model=StructuredAnswer,
            )
            assert response.structured.answer == "ok"
        finally:
            await provider.client.aclose()

    asyncio.run(run())
