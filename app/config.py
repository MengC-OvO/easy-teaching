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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
