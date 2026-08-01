from enum import Enum
from typing import Any, Dict, Optional


class ModelErrorCode(str, Enum):
    CONFIGURATION_ERROR = "configuration_error"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    RATE_LIMITED = "rate_limited"
    INVALID_RESPONSE = "invalid_response"
    PROVIDER_ERROR = "provider_error"


class ModelProviderError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: ModelErrorCode = ModelErrorCode.PROVIDER_ERROR,
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.recoverable = recoverable
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "recoverable": self.recoverable,
            "details": self.details,
        }

    def safe_metadata(self) -> Dict[str, Any]:
        """Return operational fields that are safe for traces and API events."""
        safe = {
            "code": self.code.value,
            "recoverable": self.recoverable,
        }
        for key in (
            "attempts",
            "max_attempts",
            "retry_exhausted",
            "status_code",
            "structured_attempts",
            "structured_max_attempts",
        ):
            value = self.details.get(key)
            if isinstance(value, (str, int, float, bool)):
                safe[key] = value
        return safe


class ModelConfigurationError(ModelProviderError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            code=ModelErrorCode.CONFIGURATION_ERROR,
            recoverable=False,
            details=details,
        )


class ModelTimeoutError(ModelProviderError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            code=ModelErrorCode.TIMEOUT,
            recoverable=True,
            details=details,
        )


class ModelHTTPError(ModelProviderError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_details = {"status_code": status_code}
        if details:
            merged_details.update(details)
        super().__init__(
            message,
            code=(
                ModelErrorCode.RATE_LIMITED
                if status_code == 429
                else ModelErrorCode.HTTP_ERROR
            ),
            recoverable=status_code >= 500 or status_code == 429,
            details=merged_details,
        )
        self.status_code = status_code


class ModelInvalidResponseError(ModelProviderError):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message,
            code=ModelErrorCode.INVALID_RESPONSE,
            recoverable=True,
            details=details,
        )
