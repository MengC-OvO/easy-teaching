from app.services.model_errors import ModelHTTPError
from scripts.build_vector_index import retry_after_seconds


def test_retry_after_seconds_reads_gemini_quota_message() -> None:
    error = ModelHTTPError(
        "Embedding provider returned an HTTP error.",
        status_code=429,
        details={
            "body": "Quota exceeded. Please retry in 29.316931117s.",
        },
    )

    assert retry_after_seconds(error) == 29.316931117


def test_retry_after_seconds_returns_none_when_missing() -> None:
    error = ModelHTTPError(
        "Embedding provider returned an HTTP error.",
        status_code=500,
        details={"body": "temporary backend error"},
    )

    assert retry_after_seconds(error) is None
