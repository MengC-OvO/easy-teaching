"""Local 4-bit Qwen + LoRA inference adapter with strict JSON validation."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from safety_gateway.contracts import ModelAnnotation
from safety_gateway.prompt import SYSTEM_PROMPT


class ModelInferenceError(RuntimeError):
    """Sanitized inference failure; never contains source text or model output."""


class SafetyAnnotator(Protocol):
    ready: bool
    model_loaded: bool

    async def annotate(self, text: str) -> ModelAnnotation: ...


class LocalQwenAnnotator:
    """Serialize GPU generation because one local model serves one request at a time."""

    ready = True
    model_loaded = True

    def __init__(self, *, tokenizer, model, torch_module, max_input_tokens: int, max_new_tokens: int) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._max_input_tokens = max_input_tokens
        self._max_new_tokens = max_new_tokens
        self._generation_lock = asyncio.Lock()

    @classmethod
    def load(
        cls,
        *,
        model_dir: Path,
        adapter_dir: Path,
        max_input_tokens: int = 1536,
        max_new_tokens: int = 320,
    ) -> "LocalQwenAnnotator":
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as error:
            raise ModelInferenceError("local model runtime dependencies are unavailable") from error
        if not torch.cuda.is_available():
            raise ModelInferenceError("CUDA GPU is unavailable")
        model_dir = model_dir.resolve()
        adapter_dir = adapter_dir.resolve()
        if not model_dir.is_dir() or not adapter_dir.is_dir():
            raise ModelInferenceError("local model or adapter directory is unavailable")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            quantization_config=quantization,
            dtype=compute_dtype,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            local_files_only=True,
            is_trainable=False,
        )
        model.eval()
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

    async def annotate(self, text: str) -> ModelAnnotation:
        async with self._generation_lock:
            return await asyncio.to_thread(self._annotate_sync, text)

    def _annotate_sync(self, text: str) -> ModelAnnotation:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self._max_input_tokens:
            raise ModelInferenceError("input exceeds the configured local-model token limit")
        inputs = inputs.to(self._model.device)
        try:
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            answer_tokens = generated[0, inputs["input_ids"].shape[1]:]
            answer = self._tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            return ModelAnnotation.model_validate_json(answer)
        except ModelInferenceError:
            raise
        except Exception as error:
            raise ModelInferenceError("local model returned an invalid annotation") from error
