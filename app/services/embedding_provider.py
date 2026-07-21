import json
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from app.config import Settings, settings
from app.services.model_errors import (
    ModelConfigurationError,
    ModelHTTPError,
    ModelInvalidResponseError,
    ModelProviderError,
    ModelTimeoutError,
)


class EmbeddingResponse(BaseModel):
    vectors: List[List[float]]
    model: str
    dimension: int = Field(ge=1)
    raw_response: Dict[str, Any] = Field(default_factory=dict)


class GeminiEmbeddingProvider:
    def __init__(
        self,
        provider_settings: Settings = settings,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.settings = provider_settings
        self.client = client or httpx.Client()

    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        return self.embed_texts([text], task_type=task_type).vectors[0]

    def embed_texts(
        self,
        texts: List[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> EmbeddingResponse:
        self._validate_configuration()
        if not texts:
            raise ValueError("texts must contain at least one item")

        payload = self._build_payload(texts, task_type=task_type)

        try:
            response = self.client.post(
                self._batch_embed_url(),
                headers=self._headers(),
                json=payload,
                timeout=self.settings.embedding_timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise ModelTimeoutError(
                "Embedding request timed out.",
                details={"model": self.settings.embedding_model_name},
            ) from error
        except httpx.RequestError as error:
            raise ModelProviderError(
                "Embedding request failed before receiving a response.",
                details={
                    "error": str(error),
                    "model": self.settings.embedding_model_name,
                },
            ) from error

        if response.status_code >= 400:
            raise ModelHTTPError(
                "Embedding provider returned an HTTP error.",
                status_code=response.status_code,
                details={
                    "body": response.text[:500],
                    "model": self.settings.embedding_model_name,
                },
            )

        raw_response = self._parse_response_json(response)
        vectors = self._extract_vectors(raw_response, expected_count=len(texts))
        return EmbeddingResponse(
            vectors=vectors,
            model=self.settings.embedding_model_name,
            dimension=self.settings.embedding_dimension,
            raw_response=raw_response,
        )

    def _validate_configuration(self) -> None:
        missing = []
        if not self.settings.embedding_base_url:
            missing.append("EMBEDDING_BASE_URL")
        if not self.settings.embedding_api_key:
            missing.append("EMBEDDING_API_KEY")
        if not self.settings.embedding_model_name:
            missing.append("EMBEDDING_MODEL_NAME")
        if not self.settings.embedding_dimension:
            missing.append("EMBEDDING_DIMENSION")

        if missing:
            raise ModelConfigurationError(
                "Embedding provider configuration is incomplete.",
                details={"missing": missing},
            )

    def _batch_embed_url(self) -> str:
        base_url = self.settings.embedding_base_url.rstrip("/")
        model = self.settings.embedding_model_name
        return f"{base_url}/models/{model}:batchEmbedContents"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.settings.embedding_api_key,
        }

    def _build_payload(self, texts: List[str], *, task_type: str) -> Dict[str, Any]:
        model = f"models/{self.settings.embedding_model_name}"
        return {
            "requests": [
                {
                    "model": model,
                    "taskType": task_type,
                    "outputDimensionality": self.settings.embedding_dimension,
                    "content": {
                        "parts": [{"text": text}],
                    },
                }
                for text in texts
            ]
        }

    def _parse_response_json(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            raw_response = response.json()
        except json.JSONDecodeError as error:
            raise ModelInvalidResponseError(
                "Embedding provider returned invalid JSON.",
                details={"body": response.text[:500]},
            ) from error

        if not isinstance(raw_response, dict):
            raise ModelInvalidResponseError(
                "Embedding provider response must be a JSON object.",
                details={"type": type(raw_response).__name__},
            )
        return raw_response

    def _extract_vectors(
        self,
        raw_response: Dict[str, Any],
        *,
        expected_count: int,
    ) -> List[List[float]]:
        embeddings = raw_response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            raise ModelInvalidResponseError(
                "Embedding provider response did not include the expected embeddings.",
                details={
                    "expected_count": expected_count,
                    "actual_count": len(embeddings) if isinstance(embeddings, list) else None,
                },
            )

        vectors: List[List[float]] = []
        for embedding in embeddings:
            values = embedding.get("values") if isinstance(embedding, dict) else None
            if not isinstance(values, list):
                raise ModelInvalidResponseError(
                    "Embedding values must be a list.",
                    details={"embedding_type": type(embedding).__name__},
                )
            vector = [float(value) for value in values]
            if len(vector) != self.settings.embedding_dimension:
                raise ModelInvalidResponseError(
                    "Embedding vector dimension did not match configuration.",
                    details={
                        "expected_dimension": self.settings.embedding_dimension,
                        "actual_dimension": len(vector),
                    },
                )
            vectors.append(vector)
        return vectors
