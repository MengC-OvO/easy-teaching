import pytest
from pydantic import ValidationError

from app.services import RetryPolicy


def test_retry_policy_uses_capped_exponential_delays() -> None:
    policy = RetryPolicy(
        max_attempts=5,
        initial_delay_seconds=0.5,
        max_delay_seconds=2.0,
    )

    assert [policy.delay_after(attempt) for attempt in range(1, 5)] == [
        0.5,
        1.0,
        2.0,
        2.0,
    ]


def test_retry_policy_rejects_inverted_delay_range() -> None:
    with pytest.raises(ValidationError, match="max_delay_seconds"):
        RetryPolicy(initial_delay_seconds=2.0, max_delay_seconds=1.0)
