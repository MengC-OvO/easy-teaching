"""Atomic Redis fixed-window limiter for horizontally scaled API replicas."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any


_FIXED_WINDOW_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {current, ttl}
"""


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: int


class RedisRateLimiter:
    def __init__(self, client: Any, *, limit: int, window_seconds: int) -> None:
        self.client = client
        self.limit = limit
        self.window_seconds = window_seconds

    async def check(self, identity: str) -> RateLimitDecision:
        # Do not expose a Supabase user ID or client address in Redis key names.
        identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
        key = f"easyteaching:rate:messages:{identity_hash}"
        current, ttl = await self.client.eval(
            _FIXED_WINDOW_SCRIPT,
            1,
            key,
            self.window_seconds,
        )
        current = int(current)
        ttl = max(1, int(ttl))
        return RateLimitDecision(
            allowed=current <= self.limit,
            remaining=max(0, self.limit - current),
            retry_after_seconds=ttl,
        )
