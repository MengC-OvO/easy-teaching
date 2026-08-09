import asyncio

import httpx
import pytest

from app.config import Settings
from app.services import (
    GeminiEmbeddingProvider,
    ModelConfigurationError,
    ModelHTTPError,
    ModelInvalidResponseError,
    ModelTimeoutError,
)


def make_settings() -> Settings:
    return Settings(
        EMBEDDING_BASE_URL="https://embedding.test/v1beta",
        EMBEDDING_API_KEY="test-key",
        EMBEDDING_MODEL_NAME="gemini-embedding-001",
        EMBEDDING_DIMENSION=3,
        EMBEDDING_TIMEOUT_SECONDS=20,
    )


def make_provider(handler) -> GeminiEmbeddingProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return GeminiEmbeddingProvider(make_settings(), client=client)


def test_embedding_provider_sends_gemini_batch_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "https://embedding.test/v1beta/models/"
            "gemini-embedding-001:batchEmbedContents"
        )
        assert request.headers["x-goog-api-key"] == "test-key"
        payload = request.read()
        assert b"models/gemini-embedding-001" in payload
        assert b"RETRIEVAL_DOCUMENT" in payload
        assert b"outputDimensionality" in payload
        assert b"hello" in payload
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]},
        )

    response = make_provider(handler).embed_texts(["hello"])

    assert response.model == "gemini-embedding-001"
    assert response.dimension == 3
    assert response.vectors == [[0.1, 0.2, 0.3]]


def test_embedding_provider_embed_text_uses_query_task_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b"RETRIEVAL_QUERY" in request.read()
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]},
        )

    vector = make_provider(handler).embed_text("query")

    assert vector == [0.1, 0.2, 0.3]


def test_embedding_provider_async_path_uses_async_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert b"RETRIEVAL_QUERY" in request.read()
        return httpx.Response(
            200,
            json={"embeddings": [{"values": [0.1, 0.2, 0.3]}]},
        )

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = GeminiEmbeddingProvider(make_settings(), async_client=client)
        try:
            return await provider.embed_text_async("query")
        finally:
            await client.aclose()

    assert asyncio.run(run()) == [0.1, 0.2, 0.3]


def test_embedding_provider_requires_configuration() -> None:
    provider = GeminiEmbeddingProvider(
        Settings(
            EMBEDDING_BASE_URL="",
            EMBEDDING_API_KEY="",
            EMBEDDING_MODEL_NAME="",
            EMBEDDING_DIMENSION=0,
        ),
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )

    with pytest.raises(ModelConfigurationError) as error:
        provider.embed_texts(["hello"])

    assert error.value.details["missing"] == [
        "EMBEDDING_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_MODEL_NAME",
        "EMBEDDING_DIMENSION",
    ]


def test_embedding_provider_rejects_empty_text_list() -> None:
    with pytest.raises(ValueError):
        make_provider(lambda request: httpx.Response(200)).embed_texts([])


def test_embedding_provider_wraps_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    with pytest.raises(ModelTimeoutError):
        make_provider(handler).embed_texts(["hello"])


def test_embedding_provider_wraps_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    with pytest.raises(ModelHTTPError) as error:
        make_provider(handler).embed_texts(["hello"])

    assert error.value.recoverable is True
    assert error.value.details["status_code"] == 429


def test_embedding_provider_rejects_wrong_embedding_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": []})

    with pytest.raises(ModelInvalidResponseError):
        make_provider(handler).embed_texts(["hello"])


def test_embedding_provider_rejects_wrong_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]})

    with pytest.raises(ModelInvalidResponseError) as error:
        make_provider(handler).embed_texts(["hello"])

    assert error.value.details["expected_dimension"] == 3
    assert error.value.details["actual_dimension"] == 2
