"""Token and cost accounting around the existing model provider."""

from typing import List, Type

from pydantic import BaseModel

from app.services import (
    ChatCompletionsModelProvider,
    ModelMessage,
    ModelRequest,
    ModelResponse,
)
from evals.schemas import EvalTokenUsage


class MeteredModelProvider:
    """Delegate model calls while collecting usage for the current eval case."""

    def __init__(
        self,
        provider: ChatCompletionsModelProvider,
        *,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        self.provider = provider
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.reset()

    def reset(self) -> None:
        self._usage = EvalTokenUsage()

    @property
    def usage(self) -> EvalTokenUsage:
        return self._usage.model_copy()

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self._usage.prompt_tokens * self.input_cost_per_million
            + self._usage.completion_tokens * self.output_cost_per_million
        ) / 1_000_000

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await self.provider.generate(request)
        self._record(response)
        return response

    async def generate_structured(
        self,
        *,
        messages: List[ModelMessage],
        response_model: Type[BaseModel],
        temperature: float = 0.0,
    ) -> ModelResponse:
        response = await self.provider.generate_structured(
            messages=messages,
            response_model=response_model,
            temperature=temperature,
        )
        self._record(response)
        return response

    def _record(self, response: ModelResponse) -> None:
        prompt = response.usage.prompt_tokens or 0
        completion = response.usage.completion_tokens or 0
        reported_total = response.usage.total_tokens or 0
        self._usage = EvalTokenUsage(
            model_calls=self._usage.model_calls + 1,
            prompt_tokens=self._usage.prompt_tokens + prompt,
            completion_tokens=self._usage.completion_tokens + completion,
            total_tokens=self._usage.total_tokens
            + max(reported_total, prompt + completion),
        )
