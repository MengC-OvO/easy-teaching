from app.config import Settings


def test_model_settings_have_safe_defaults() -> None:
    settings = Settings()

    assert settings.model_chat_completions_path == "/chat/completions"
    assert settings.model_name == "gemini-2.5-flash"
    assert settings.model_timeout_seconds == 10.0


def test_env_example_does_not_contain_real_model_api_key() -> None:
    with open(".env.example", encoding="utf-8") as file:
        env_example = file.read()

    assert "MODEL_API_KEY=replace-with-your-local-key" in env_example
    assert "AQ." not in env_example
