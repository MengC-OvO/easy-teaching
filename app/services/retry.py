"""Small, reusable retry policy for transient external-service failures."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetryPolicy(BaseModel):
    """Bound retry count, exponential delays, and total wall-clock time."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_delay_seconds: float = Field(default=0.5, ge=0.0, le=60.0)
    max_delay_seconds: float = Field(default=2.0, ge=0.0, le=300.0)
    total_timeout_seconds: float = Field(default=30.0, gt=0.0, le=600.0)

    @model_validator(mode="after")
    def validate_delay_range(self) -> "RetryPolicy":
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError(
                "max_delay_seconds cannot be less than initial_delay_seconds"
            )
        return self

    def delay_after(self, failed_attempt: int) -> float:
        """Return 0.5, 1, 2... style delay, capped at the configured maximum."""
        if failed_attempt < 1:
            raise ValueError("failed_attempt must be at least 1")
        return min(
            self.initial_delay_seconds * (2 ** (failed_attempt - 1)),
            self.max_delay_seconds,
        )
