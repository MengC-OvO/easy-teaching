from evals import safe_eval_error_code
from app.services import (
    ModelHTTPError,
    ModelInvalidResponseError,
    ModelTimeoutError,
)


def test_eval_error_codes_distinguish_model_failure_classes() -> None:
    assert safe_eval_error_code(ModelTimeoutError("private")) == "timeout"
    assert (
        safe_eval_error_code(ModelHTTPError("private", status_code=429))
        == "rate_limited"
    )
    assert (
        safe_eval_error_code(ModelInvalidResponseError("private"))
        == "invalid_response"
    )


def test_eval_error_code_hides_unexpected_exception_details() -> None:
    error = RuntimeError("secret provider response")

    assert safe_eval_error_code(error) == "execution_error"
