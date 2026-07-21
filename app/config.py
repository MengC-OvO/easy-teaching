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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
