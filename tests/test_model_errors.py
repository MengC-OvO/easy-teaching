from app.services import (
    ModelConfigurationError,
    ModelErrorCode,
    ModelHTTPError,
    ModelInvalidResponseError,
    ModelProviderError,
    ModelTimeoutError,
)


def test_model_provider_error_serializes_to_dict() -> None:
    error = ModelProviderError(
        "Provider failed.",
        details={"provider": "chat_completions"},
    )

    assert error.to_dict() == {
        "code": "provider_error",
        "message": "Provider failed.",
        "recoverable": True,
        "details": {"provider": "chat_completions"},
    }


def test_configuration_error_is_not_recoverable() -> None:
    error = ModelConfigurationError("MODEL_API_KEY is missing.")

    assert error.code is ModelErrorCode.CONFIGURATION_ERROR
    assert error.recoverable is False


def test_timeout_error_is_recoverable() -> None:
    error = ModelTimeoutError("Model request timed out.")

    assert error.code is ModelErrorCode.TIMEOUT
    assert error.recoverable is True


def test_http_error_recoverability_depends_on_status_code() -> None:
    rate_limited = ModelHTTPError("Rate limited.", status_code=429)
    bad_request = ModelHTTPError("Bad request.", status_code=400)
    server_error = ModelHTTPError("Server error.", status_code=503)

    assert rate_limited.recoverable is True
    assert bad_request.recoverable is False
    assert server_error.recoverable is True
    assert server_error.details["status_code"] == 503


def test_invalid_response_error_is_recoverable() -> None:
    error = ModelInvalidResponseError("Response did not match schema.")

    assert error.code is ModelErrorCode.INVALID_RESPONSE
    assert error.recoverable is True
