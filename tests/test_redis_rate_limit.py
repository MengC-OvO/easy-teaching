"""Execute the production Lua script; install fakeredis[lua] to run locally."""

import asyncio
import time

import pytest

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lupa")

from app.services.redis_rate_limit import RedisRateLimiter


@pytest.fixture
def clock(monkeypatch):
    now = [1_700_000_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    return now


def test_sliding_boundary_releases_only_expired_requests(clock):
    async def scenario():
        async with fakeredis.aioredis.FakeRedis() as redis:
            limiter = RedisRateLimiter(redis, limit=2, window_seconds=60)
            assert (await limiter.check("a")).allowed
            clock[0] += 59
            assert (await limiter.check("a")).allowed
            denied = await limiter.check("a")
            assert not denied.allowed and denied.retry_after_seconds == 1
            clock[0] += 1
            assert (await limiter.check("a")).allowed
            denied = await limiter.check("a")
            assert not denied.allowed and denied.retry_after_seconds == 59
            key = (await redis.keys())[0]
            assert await redis.zcard(key) == 2
    asyncio.run(scenario())


def test_shared_clients_same_millisecond_and_identity_isolation(clock):
    async def scenario():
        server = fakeredis.FakeServer()
        async with fakeredis.aioredis.FakeRedis(server=server) as a, fakeredis.aioredis.FakeRedis(server=server) as b:
            limiters = [RedisRateLimiter(c, limit=20, window_seconds=60) for c in (a, b)]
            decisions = await asyncio.gather(*(limiters[i % 2].check("a") for i in range(100)))
            assert sum(d.allowed for d in decisions) == 20
            assert (await limiters[1].check("b")).allowed
            assert sorted([await a.zcard(k) for k in await a.keys()]) == [1, 20]
    asyncio.run(scenario())


def test_rejected_requests_do_not_extend_expiry_and_retry_rounds_up(clock):
    async def scenario():
        async with fakeredis.aioredis.FakeRedis() as redis:
            limiter = RedisRateLimiter(redis, limit=1, window_seconds=60)
            await limiter.check("a")
            clock[0] += 59.25
            denied = await limiter.check("a")
            assert not denied.allowed and denied.retry_after_seconds == 1
            key = (await redis.keys())[0]
            assert 0 < await redis.pttl(key) <= 751
            clock[0] += 0.75
            assert (await limiter.check("a")).allowed
    asyncio.run(scenario())


def test_lowered_limit_waits_until_enough_entries_expire(clock):
    async def scenario():
        async with fakeredis.aioredis.FakeRedis() as redis:
            limiter = RedisRateLimiter(redis, limit=3, window_seconds=60)
            for _ in range(3):
                await limiter.check("a")
                clock[0] += 10
            stricter = RedisRateLimiter(redis, limit=1, window_seconds=60)
            denied = await stricter.check("a")
            assert not denied.allowed and denied.retry_after_seconds == 50
    asyncio.run(scenario())


def test_legacy_string_key_does_not_conflict(clock):
    async def scenario():
        async with fakeredis.aioredis.FakeRedis() as redis:
            import hashlib
            digest = hashlib.sha256(b"a").hexdigest()[:32]
            legacy = f"easyteaching:rate:messages:{digest}"
            await redis.set(legacy, "20", ex=60)
            limiter = RedisRateLimiter(redis, limit=20, window_seconds=60)
            assert (await limiter.check("a")).allowed
            assert await redis.get(legacy) == b"20"
    asyncio.run(scenario())
