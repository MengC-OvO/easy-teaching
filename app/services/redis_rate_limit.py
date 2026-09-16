"""Atomic Redis sliding-window log limiter shared by API replicas."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any
from uuid import uuid4


_SLIDING_WINDOW_SCRIPT = """
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local clock = redis.call('TIME')
local now = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)

-- Keep admitted requests in (now - window_ms, now].
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window_ms)
local current = redis.call('ZCARD', KEYS[1])
if current >= limit then
  -- Also handle a limit lowered while existing entries are still live.
  local oldest = redis.call('ZRANGE', KEYS[1], current - limit, current - limit, 'WITHSCORES')
  local retry_ms = tonumber(oldest[2]) + window_ms - now
  return {0, 0, math.max(1, math.ceil(retry_ms / 1000))}
end

redis.call('ZADD', KEYS[1], now, ARGV[3])
redis.call('PEXPIRE', KEYS[1], window_ms)
return {1, limit - current - 1, 0}
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
        # Version the key to avoid WRONGTYPE against old String counters.
        key = f"easyteaching:rate:messages:sliding:v1:{identity_hash}"
        allowed, remaining, retry_after = await self.client.eval(
            _SLIDING_WINDOW_SCRIPT,
            1,
            key,
            self.window_seconds * 1000,
            self.limit,
            uuid4().hex,
        )
        return RateLimitDecision(
            allowed=bool(int(allowed)),
            remaining=int(remaining),
            retry_after_seconds=int(retry_after),
        )
