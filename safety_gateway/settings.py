"""Environment settings owned by the standalone safety gateway."""
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_backend: Literal["auto", "cuda", "mps"] = Field(
        default="auto", validation_alias="SAFETY_MODEL_BACKEND"
    )
    model_dir: Path = Field(
        default=Path("local_models/Qwen2.5-3B-Instruct"),
        validation_alias="SAFETY_MODEL_DIR",
    )
    adapter_dir: Path = Field(
        default=Path("local_models/easyteaching-safety-lora"),
        validation_alias="SAFETY_ADAPTER_DIR",
    )
    max_input_tokens: int = Field(
        default=1536, ge=128, le=8192, validation_alias="SAFETY_MAX_INPUT_TOKENS"
    )
    max_new_tokens: int = Field(
        default=320, ge=64, le=1024, validation_alias="SAFETY_MAX_NEW_TOKENS"
    )
    mapping_ttl_seconds: int = Field(
        default=3600, ge=60, le=86_400, validation_alias="SAFETY_MAPPING_TTL_SECONDS"
    )

    model_config = SettingsConfigDict(env_file=(".env", ".safety-gateway.env"), extra="ignore")
