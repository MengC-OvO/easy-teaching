from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="easy-teaching", validation_alias="APP_NAME")
    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    privacy_gateway_mode: Literal["disabled", "shadow", "enforce"] = Field(
        default="disabled",
        validation_alias="PRIVACY_GATEWAY_MODE",
    )
    privacy_gateway_url: str = Field(
        default="http://127.0.0.1:8010",
        validation_alias="PRIVACY_GATEWAY_URL",
    )
    privacy_gateway_timeout_seconds: float = Field(
        default=15.0, gt=0.0, validation_alias="PRIVACY_GATEWAY_TIMEOUT_SECONDS"
    )
    database_url: str = Field(default="", validation_alias="DATABASE_URL")
    checkpoint_database_url: str = Field(
        default="",
        validation_alias="CHECKPOINT_DATABASE_URL",
    )
    model_base_url: str = Field(default="", validation_alias="MODEL_BASE_URL")
    model_chat_completions_path: str = Field(
        default="/chat/completions",
        validation_alias="MODEL_CHAT_COMPLETIONS_PATH",
    )
    model_api_key: str = Field(default="", validation_alias="MODEL_API_KEY")
    model_name: str = Field(default="gemini-2.5-flash", validation_alias="MODEL_NAME")
    model_timeout_seconds: float = Field(
        default=10.0,
        validation_alias="MODEL_TIMEOUT_SECONDS",
    )
    model_retry_max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        validation_alias="MODEL_RETRY_MAX_ATTEMPTS",
    )
    model_retry_initial_delay_seconds: float = Field(
        default=0.5,
        ge=0.0,
        validation_alias="MODEL_RETRY_INITIAL_DELAY_SECONDS",
    )
    model_retry_max_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        validation_alias="MODEL_RETRY_MAX_DELAY_SECONDS",
    )
    model_total_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        validation_alias="MODEL_TOTAL_TIMEOUT_SECONDS",
    )
    model_structured_max_attempts: int = Field(
        default=3,
        ge=1,
        le=3,
        validation_alias="MODEL_STRUCTURED_MAX_ATTEMPTS",
    )
    embedding_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        validation_alias="EMBEDDING_BASE_URL",
    )
    embedding_api_key: str = Field(default="", validation_alias="EMBEDDING_API_KEY")
    embedding_model_name: str = Field(
        default="gemini-embedding-001",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dimension: int = Field(default=768, validation_alias="EMBEDDING_DIMENSION")
    embedding_timeout_seconds: float = Field(
        default=20.0,
        validation_alias="EMBEDDING_TIMEOUT_SECONDS",
    )
    chroma_path: str = Field(
        default="data/chroma",
        validation_alias="CHROMA_PATH",
    )
    chroma_collection_name: str = Field(
        default="easyteaching_knowledge",
        validation_alias="CHROMA_COLLECTION_NAME",
    )
    lexical_index_path: str = Field(
        default="data/knowledge/index/knowledge_fts.sqlite3",
        validation_alias="LEXICAL_INDEX_PATH",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RERANKER_MODEL_NAME",
    )
    auth_enabled: bool = Field(default=False, validation_alias="AUTH_ENABLED")
    supabase_url: str = Field(default="", validation_alias="SUPABASE_URL")
    supabase_publishable_key: str = Field(
        default="",
        validation_alias="SUPABASE_PUBLISHABLE_KEY",
    )
    auth_timeout_seconds: float = Field(
        default=5.0,
        gt=0.0,
        validation_alias="AUTH_TIMEOUT_SECONDS",
    )
    auth_cookie_max_age_seconds: int = Field(
        default=3600,
        ge=60,
        validation_alias="AUTH_COOKIE_MAX_AGE_SECONDS",
    )
    google_drive_mcp_enabled: bool = Field(
        default=False,
        validation_alias="GOOGLE_DRIVE_MCP_ENABLED",
    )
    google_drive_mcp_command: str = Field(
        default="workspace-mcp",
        validation_alias="GOOGLE_DRIVE_MCP_COMMAND",
    )
    google_drive_user_email: str = Field(
        default="",
        validation_alias="GOOGLE_DRIVE_USER_EMAIL",
    )
    google_drive_mcp_timeout_seconds: float = Field(
        default=45.0,
        gt=0.0,
        validation_alias="GOOGLE_DRIVE_MCP_TIMEOUT_SECONDS",
    )
    google_oauth_client_id: str = Field(
        default="",
        validation_alias="GOOGLE_OAUTH_CLIENT_ID",
    )
    google_oauth_client_secret: str = Field(
        default="",
        validation_alias="GOOGLE_OAUTH_CLIENT_SECRET",
    )
    google_workspace_mcp_credentials_dir: str = Field(
        default="data/local/google_workspace_mcp",
        validation_alias="WORKSPACE_MCP_CREDENTIALS_DIR",
    )
    upload_root: str = Field(
        default="data/local/uploads", validation_alias="UPLOAD_ROOT"
    )
    upload_max_bytes: int = Field(
        default=15 * 1024 * 1024, ge=1, validation_alias="UPLOAD_MAX_BYTES"
    )
    scoped_knowledge_root: str = Field(
        default="data/local/knowledge/tenants",
        validation_alias="SCOPED_KNOWLEDGE_ROOT",
    )
    official_web_search_enabled: bool = Field(
        default=False, validation_alias="OFFICIAL_WEB_SEARCH_ENABLED"
    )
    official_web_search_api_key: str = Field(
        default="", validation_alias="OFFICIAL_WEB_SEARCH_API_KEY"
    )
    official_web_search_engine_id: str = Field(
        default="", validation_alias="OFFICIAL_WEB_SEARCH_ENGINE_ID"
    )
    official_web_search_timeout_seconds: float = Field(
        default=12.0, gt=0, validation_alias="OFFICIAL_WEB_SEARCH_TIMEOUT_SECONDS"
    )
    voice_transcription_enabled: bool = Field(
        default=False, validation_alias="VOICE_TRANSCRIPTION_ENABLED"
    )
    whisper_model_name: str = Field(
        default="small.en", validation_alias="WHISPER_MODEL_NAME"
    )
    whisper_device: str = Field(default="auto", validation_alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(
        default="int8_float16", validation_alias="WHISPER_COMPUTE_TYPE"
    )

    @model_validator(mode="after")
    def validate_privacy_gateway_boundary(self) -> "Settings":
        if self.privacy_gateway_mode == "shadow" and self.app_env.lower() not in {
            "local",
            "test",
            "development",
        }:
            raise ValueError("shadow privacy mode is restricted to local synthetic diagnostics")
        if self.privacy_gateway_mode != "disabled":
            parsed = urlparse(self.privacy_gateway_url)
            if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ValueError("privacy gateway must use a loopback HTTP(S) URL")
        if self.google_drive_mcp_enabled:
            if not self.google_drive_user_email:
                raise ValueError(
                    "GOOGLE_DRIVE_USER_EMAIL is required when Drive MCP is enabled"
                )
            if not self.google_oauth_client_id or not self.google_oauth_client_secret:
                raise ValueError(
                    "Google OAuth client credentials are required when Drive MCP is enabled"
                )
        if self.official_web_search_enabled and (
            not self.official_web_search_api_key
            or not self.official_web_search_engine_id
        ):
            raise ValueError(
                "Official web search requires its API key and engine ID when enabled"
            )
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
