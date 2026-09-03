#!/usr/bin/env python3
"""Live Redis Stream smoke/load check using an isolated disposable run key."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from redis.asyncio import Redis  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.redis_event_bus import RedisEventBus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the live Redis progress stream.")
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    return parser.parse_args()


async def run(events: int, concurrency: int) -> dict:
    if events < 1 or concurrency < 1:
        raise ValueError("events and concurrency must be positive")
    request_id = f"live-progress-{uuid4()}"
    client = Redis.from_url(
        settings.redis_progress_url or settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=20,
    )
    bus = RedisEventBus(
        client,
        maxlen=max(settings.redis_progress_stream_maxlen, events),
        ttl_seconds=settings.redis_progress_ttl_seconds,
    )
    stream_key = bus._stream_key(request_id)
    sequence_key = bus._sequence_key(request_id)
    semaphore = asyncio.Semaphore(concurrency)

    async def publish(index: int) -> None:
        async with semaphore:
            await bus.publish(
                request_id=request_id,
                session_id="live-test-session",
                event="trace",
                data={"step": "synthetic", "index": index},
            )

    try:
        if not await client.ping():
            raise ConnectionError("Redis ping failed")
        started = perf_counter()
        await asyncio.gather(*(publish(index) for index in range(events)))
        elapsed = perf_counter() - started

        received = []
        cursor = "0-0"
        while len(received) < events:
            batch = await bus.read(
                request_id,
                after_event_id=cursor,
                block_ms=1000,
                count=min(200, events - len(received)),
            )
            if not batch:
                break
            received.extend(batch)
            cursor = batch[-1].event_id

        sequences = [item.sequence for item in received]
        if sequences != list(range(1, events + 1)):
            raise AssertionError("Redis Stream sequence or replay order is incorrect")
        return {
            "events": events,
            "concurrency": concurrency,
            "elapsed_seconds": round(elapsed, 4),
            "publish_events_per_second": round(events / elapsed, 2),
            "replayed_events": len(received),
            "stream_length": await client.xlen(stream_key),
            "ttl_seconds": await client.ttl(stream_key),
            "ordered": True,
        }
    finally:
        await client.delete(stream_key, sequence_key)
        await client.aclose()


def main() -> int:
    args = parse_args()
    print(json.dumps(asyncio.run(run(args.events, args.concurrency)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
