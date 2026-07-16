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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
