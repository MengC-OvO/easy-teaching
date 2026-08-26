import pytest

from app.config import Settings


def test_model_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.privacy_gateway_mode == "disabled"
    assert settings.privacy_gateway_url == "http://127.0.0.1:8010"
    assert settings.privacy_gateway_timeout_seconds == 15.0
    assert settings.model_chat_completions_path == "/chat/completions"
    assert settings.model_name == "gemini-2.5-flash"
    assert settings.model_timeout_seconds == 10.0
    assert settings.model_retry_max_attempts == 3
    assert settings.model_retry_initial_delay_seconds == 0.5
    assert settings.model_retry_max_delay_seconds == 2.0
    assert settings.model_total_timeout_seconds == 30.0
    assert settings.model_structured_max_attempts == 3
    assert settings.embedding_base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert settings.embedding_model_name == "gemini-embedding-001"
    assert settings.embedding_dimension == 768
    assert settings.embedding_timeout_seconds == 20.0
    assert settings.chroma_path == "data/chroma"
    assert settings.chroma_collection_name == "easyteaching_knowledge"
    assert settings.database_url == ""
    assert settings.checkpoint_database_url == ""
    assert settings.auth_enabled is False
    assert settings.supabase_url == ""
    assert settings.supabase_publishable_key == ""


def test_embedding_settings_can_be_overridden() -> None:
    settings = Settings(
        EMBEDDING_BASE_URL="https://example.test/v1",
        EMBEDDING_API_KEY="test-key",
        EMBEDDING_MODEL_NAME="test-embedding-model",
        EMBEDDING_DIMENSION="1536",
        EMBEDDING_TIMEOUT_SECONDS="30",
        CHROMA_PATH="tmp/chroma",
        CHROMA_COLLECTION_NAME="test_collection",
    )

    assert settings.embedding_base_url == "https://example.test/v1"
    assert settings.embedding_api_key == "test-key"
    assert settings.embedding_model_name == "test-embedding-model"
    assert settings.embedding_dimension == 1536
    assert settings.embedding_timeout_seconds == 30.0
    assert settings.chroma_path == "tmp/chroma"
    assert settings.chroma_collection_name == "test_collection"


def test_env_example_does_not_contain_real_model_api_key() -> None:
    with open(".env.example", encoding="utf-8") as file:
        env_example = file.read()

    assert "MODEL_API_KEY=replace-with-your-local-key" in env_example
    assert "EMBEDDING_API_KEY=replace-with-your-local-key" in env_example
    assert "AQ." not in env_example


def test_shadow_mode_is_rejected_outside_local_diagnostics() -> None:
    with pytest.raises(ValueError, match="shadow privacy mode"):
        Settings(
            _env_file=None,
            APP_ENV="production",
            PRIVACY_GATEWAY_MODE="shadow",
        )


def test_enabled_gateway_must_stay_on_loopback() -> None:
    with pytest.raises(ValueError, match="loopback"):
        Settings(
            _env_file=None,
            PRIVACY_GATEWAY_MODE="enforce",
            PRIVACY_GATEWAY_URL="https://remote.example.test",
        )
