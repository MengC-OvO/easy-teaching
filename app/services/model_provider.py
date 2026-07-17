import json
from typing import Any, Dict, Optional, Type

import httpx
from pydantic import BaseModel, ValidationError

from app.config import Settings, settings
from app.services.model_errors import (
    ModelConfigurationError,
    ModelHTTPError,
    ModelInvalidResponseError,
    ModelProviderError,
    ModelTimeoutError,
)
from app.services.model_types import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)


class ChatCompletionsModelProvider:
    def __init__(
        self,
        provider_settings: Settings = settings,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.settings = provider_settings
        self.client = client or httpx.Client()

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._validate_configuration()
        payload = self._build_payload(request)

        try:
            response = self.client.post(
                self._chat_completions_url(),
                headers=self._headers(),
                json=payload,
                timeout=request.timeout_seconds or self.settings.model_timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise ModelTimeoutError(
                "Model request timed out.",
                details={"model": payload["model"]},
            ) from error
        except httpx.RequestError as error:
            raise ModelProviderError(
                "Model request failed before receiving a response.",
                details={"error": str(error), "model": payload["model"]},
            ) from error

        if response.status_code >= 400:
            raise ModelHTTPError(
                "Model provider returned an HTTP error.",
                status_code=response.status_code,
                details={"body": response.text[:500], "model": payload["model"]},
            )

        raw_response = self._parse_response_json(response)
        content = self._extract_content(raw_response)
        structured = None
        if request.response_model is not None:
            structured = self._parse_structured_content(
                content,
                request.response_model,
            )

        return ModelResponse(
            content=content,
            model=raw_response.get("model"),
            finish_reason=self._extract_finish_reason(raw_response),
            usage=self._extract_usage(raw_response),
            raw_response=raw_response,
            structured=structured,
        )

    def generate_structured(
        self,
        *,
        messages: list[ModelMessage],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: Optional[float] = None,
    ) -> ModelResponse:
        return self.generate(
            ModelRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                response_model=response_model,
            )
        )

    def _validate_configuration(self) -> None:
        missing = []
        if not self.settings.model_base_url:
            missing.append("MODEL_BASE_URL")
        if not self.settings.model_api_key:
            missing.append("MODEL_API_KEY")
        if not self.settings.model_name:
            missing.append("MODEL_NAME")

        if missing:
            raise ModelConfigurationError(
                "Model provider configuration is incomplete.",
                details={"missing": missing},
            )

    def _chat_completions_url(self) -> str:
        return (
            self.settings.model_base_url.rstrip("/")
            + "/"
            + self.settings.model_chat_completions_path.lstrip("/")
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.model_api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: ModelRequest) -> Dict[str, Any]:
        return {
            "model": request.model or self.settings.model_name,
            "messages": [self._message_to_payload(message) for message in request.messages],
            "temperature": request.temperature,
        }

    def _message_to_payload(self, message: ModelMessage) -> Dict[str, str]:
        return {"role": message.role.value, "content": message.content}

    def _parse_response_json(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            raw_response = response.json()
        except json.JSONDecodeError as error:
            raise ModelInvalidResponseError(
                "Model provider returned invalid JSON.",
                details={"body": response.text[:500]},
            ) from error

        if not isinstance(raw_response, dict):
            raise ModelInvalidResponseError(
                "Model provider response must be a JSON object.",
                details={"type": type(raw_response).__name__},
            )
        return raw_response

    def _extract_content(self, raw_response: Dict[str, Any]) -> str:
        try:
            content = raw_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ModelInvalidResponseError(
                "Model provider response did not include choices[0].message.content.",
                details={"response_keys": list(raw_response.keys())},
            ) from error

        if not isinstance(content, str) or not content:
            raise ModelInvalidResponseError(
                "Model provider message content must be a non-empty string.",
                details={"content_type": type(content).__name__},
            )
        return content

    def _extract_finish_reason(self, raw_response: Dict[str, Any]) -> Optional[str]:
        try:
            finish_reason = raw_response["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            return None
        return finish_reason if isinstance(finish_reason, str) else None

    def _extract_usage(self, raw_response: Dict[str, Any]) -> ModelUsage:
        usage = raw_response.get("usage")
        if not isinstance(usage, dict):
            return ModelUsage()
        return ModelUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    def _parse_structured_content(
        self,
        content: str,
        response_model: Type[BaseModel],
    ) -> BaseModel:
        normalized_content = self._normalize_json_content(content)
        try:
            parsed = json.loads(normalized_content)
        except json.JSONDecodeError as error:
            raise ModelInvalidResponseError(
                "Model response content was not valid JSON.",
                details={"content": content[:500]},
            ) from error

        try:
            return response_model.model_validate(parsed)
        except ValidationError as error:
            raise ModelInvalidResponseError(
                "Model response JSON did not match the requested schema.",
                details={"errors": error.errors()},
            ) from error

    def _normalize_json_content(self, content: str) -> str:
        stripped = content.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped
