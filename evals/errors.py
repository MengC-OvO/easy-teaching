"""Map internal exceptions to stable, non-sensitive evaluation error codes."""

from app.services import ModelProviderError


def safe_eval_error_code(error: Exception) -> str:
    """Expose a useful category without serializing exception text or details."""
    if isinstance(error, ModelProviderError):
        return error.code.value
    if isinstance(error, TimeoutError):
        return "timeout"
    return "execution_error"
