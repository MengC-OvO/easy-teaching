from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="eduflow-au-agent", validation_alias="APP_NAME")
    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    database_path: str = Field(
        default="data/local/eduflow.sqlite3",
        validation_alias="DATABASE_PATH",
    )
    database_url: str = Field(default="", validation_alias="DATABASE_URL")
    checkpoint_database_path: str = Field(
        default="data/local/checkpoints.sqlite3",
        validation_alias="CHECKPOINT_DATABASE_PATH",
    )
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
        default=2,
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
        default="eduflow_knowledge",
        validation_alias="CHROMA_COLLECTION_NAME",
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
