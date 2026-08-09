import json
import asyncio
import time
from typing import Any, Callable, Dict, Optional, Type

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
from app.services.retry import RetryPolicy


class ChatCompletionsModelProvider:
    def __init__(
        self,
        provider_settings: Settings = settings,
        client: Optional[httpx.AsyncClient] = None,
        retry_policy: Optional[RetryPolicy] = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = provider_settings
        self.client = client or httpx.AsyncClient()
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=provider_settings.model_retry_max_attempts,
            initial_delay_seconds=(
                provider_settings.model_retry_initial_delay_seconds
            ),
            max_delay_seconds=provider_settings.model_retry_max_delay_seconds,
            total_timeout_seconds=provider_settings.model_total_timeout_seconds,
        )
        self._sleep = sleep
        self._clock = clock

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return await self._generate_with_deadline(request, started_at=self._clock())

    async def _generate_with_deadline(
        self,
        request: ModelRequest,
        *,
        started_at: float,
    ) -> ModelResponse:
        self._validate_configuration()
        payload = self._build_payload(request)
        response = await self._post_with_retry(request, payload, started_at=started_at)

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

    async def _post_with_retry(
        self,
        request: ModelRequest,
        payload: Dict[str, Any],
        *,
        started_at: float,
    ) -> httpx.Response:
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            remaining_seconds = self._remaining_seconds(started_at)
            attempt_timeout = min(
                request.timeout_seconds or self.settings.model_timeout_seconds,
                remaining_seconds,
            )
            try:
                response = await self.client.post(
                    self._chat_completions_url(),
                    headers=self._headers(),
                    json=payload,
                    timeout=attempt_timeout,
                )
                if response.status_code < 400:
                    return response
                error: ModelProviderError = ModelHTTPError(
                    "Model provider returned an HTTP error.",
                    status_code=response.status_code,
                    details={
                        "body": response.text[:500],
                        "model": payload["model"],
                    },
                )
            except httpx.TimeoutException as cause:
                error = ModelTimeoutError(
                    "Model request timed out.",
                    details={"model": payload["model"]},
                )
                error.__cause__ = cause
            except httpx.RequestError as cause:
                error = ModelProviderError(
                    "Model request failed before receiving a response.",
                    details={"error": str(cause), "model": payload["model"]},
                )
                error.__cause__ = cause

            if not self._should_retry(error, attempt, started_at):
                error.details.update(
                    {
                        "attempts": attempt,
                        "max_attempts": self.retry_policy.max_attempts,
                        "retry_exhausted": error.recoverable,
                    }
                )
                raise error

            await self._sleep(self.retry_policy.delay_after(attempt))

        raise RuntimeError("unreachable retry loop")

    def _should_retry(
        self,
        error: ModelProviderError,
        attempt: int,
        started_at: float,
    ) -> bool:
        if not error.recoverable or attempt >= self.retry_policy.max_attempts:
            return False
        delay = self.retry_policy.delay_after(attempt)
        return delay < self._remaining_seconds(started_at)

    def _remaining_seconds(self, started_at: float) -> float:
        remaining = self.retry_policy.total_timeout_seconds - (
            self._clock() - started_at
        )
        return max(remaining, 0.000_001)

    async def generate_structured(
        self,
        *,
        messages: list[ModelMessage],
        response_model: Type[BaseModel],
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout_seconds: Optional[float] = None,
    ) -> ModelResponse:
        started_at = self._clock()
        for attempt in range(1, self.settings.model_structured_max_attempts + 1):
            try:
                return await self._generate_with_deadline(
                    ModelRequest(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        timeout_seconds=timeout_seconds,
                        response_model=response_model,
                    ),
                    started_at=started_at,
                )
            except ModelInvalidResponseError as error:
                error.details.update(
                    {
                        "structured_attempts": attempt,
                        "structured_max_attempts": (
                            self.settings.model_structured_max_attempts
                        ),
                    }
                )
                if attempt >= self.settings.model_structured_max_attempts:
                    raise
        raise RuntimeError("unreachable structured-output retry loop")

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
