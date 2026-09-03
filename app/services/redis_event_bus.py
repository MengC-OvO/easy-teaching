"""Short-lived, replayable Agent progress events backed by Redis Streams."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List


_PUBLISH_SCRIPT = """
local sequence = redis.call('INCR', KEYS[2])
local event_id = redis.call(
  'XADD', KEYS[1], 'MAXLEN', '~', ARGV[1], '*',
  'event', ARGV[3],
  'sequence', tostring(sequence),
  'session_id', ARGV[4],
  'request_id', ARGV[5],
  'data', ARGV[6]
)
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[2])
return {event_id, sequence}
"""


@dataclass(frozen=True)
class ProgressEvent:
    event_id: str
    event: str
    sequence: int
    session_id: str
    request_id: str
    data: Dict[str, Any]


class RedisEventBus:
    """Publish safe progress only; final business data never belongs here."""

    def __init__(self, client: Any, *, maxlen: int, ttl_seconds: int) -> None:
        self.client = client
        self.maxlen = maxlen
        self.ttl_seconds = ttl_seconds

    async def publish(
        self,
        *,
        request_id: str,
        session_id: str,
        event: str,
        data: Dict[str, Any],
    ) -> str:
        stream_key = self._stream_key(request_id)
        sequence_key = self._sequence_key(request_id)
        event_id, _ = await self.client.eval(
            _PUBLISH_SCRIPT,
            2,
            stream_key,
            sequence_key,
            self.maxlen,
            self.ttl_seconds,
            event,
            session_id,
            request_id,
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        )
        return str(event_id)

    async def read(
        self,
        request_id: str,
        *,
        after_event_id: str,
        block_ms: int,
        count: int = 50,
    ) -> List[ProgressEvent]:
        response = await self.client.xread(
            {self._stream_key(request_id): after_event_id},
            count=count,
            block=block_ms,
        )
        events: List[ProgressEvent] = []
        for _, records in response:
            for event_id, fields in records:
                try:
                    data = json.loads(fields.get("data", "{}"))
                    if not isinstance(data, dict):
                        data = {}
                    events.append(
                        ProgressEvent(
                            event_id=str(event_id),
                            event=str(fields["event"]),
                            sequence=int(fields["sequence"]),
                            session_id=str(fields["session_id"]),
                            request_id=str(fields["request_id"]),
                            data=data,
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    # A malformed progress record is disposable and must not
                    # terminate the user's SSE connection.
                    continue
        return events

    @staticmethod
    def _stream_key(request_id: str) -> str:
        tag = RedisEventBus._key_tag(request_id)
        return f"easyteaching:run:{{{tag}}}:events"

    @staticmethod
    def _sequence_key(request_id: str) -> str:
        tag = RedisEventBus._key_tag(request_id)
        return f"easyteaching:run:{{{tag}}}:sequence"

    @staticmethod
    def _key_tag(request_id: str) -> str:
        return hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:32]
