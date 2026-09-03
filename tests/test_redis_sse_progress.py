import asyncio
from collections import defaultdict
from types import SimpleNamespace

from app.api.execution import _invoke_graph_with_progress, _publish_graph_trace
from app.api.routes.events import _redis_event_stream
from app.schemas import GraphState, StreamEventType
from app.services.redis_event_bus import ProgressEvent, RedisEventBus


def _id_tuple(value):
    return tuple(int(part) for part in str(value).split("-", 1))


class FakeStreamRedis:
    def __init__(self):
        self.counters = defaultdict(int)
        self.streams = defaultdict(list)
        self.expirations = {}
        self.next_id = 0

    async def eval(
        self, script, key_count, stream_key, sequence_key, maxlen, ttl,
        event, session_id, request_id, data
    ):
        assert "XADD" in script and key_count == 2
        self.counters[sequence_key] += 1
        sequence = self.counters[sequence_key]
        self.next_id += 1
        event_id = f"1000-{self.next_id}"
        self.streams[stream_key].append(
            (
                event_id,
                {
                    "event": event,
                    "sequence": str(sequence),
                    "session_id": session_id,
                    "request_id": request_id,
                    "data": data,
                },
            )
        )
        self.streams[stream_key] = self.streams[stream_key][-int(maxlen):]
        self.expirations[stream_key] = int(ttl)
        self.expirations[sequence_key] = int(ttl)
        return [event_id, sequence]

    async def xread(self, streams, *, count, block):
        key, cursor = next(iter(streams.items()))
        records = [
            item
            for item in self.streams[key]
            if _id_tuple(item[0]) > _id_tuple(cursor)
        ][:count]
        return [] if not records else [(key, records)]


def test_redis_stream_replays_from_event_id_and_expires_short_lived_progress() -> None:
    redis = FakeStreamRedis()
    bus = RedisEventBus(redis, maxlen=10, ttl_seconds=3600)

    async def scenario():
        first_id = await bus.publish(
            request_id="request-1",
            session_id="session-1",
            event="trace",
            data={"message": "Preparing"},
        )
        await bus.publish(
            request_id="request-1",
            session_id="session-1",
            event="completed",
            data={"status": "completed"},
        )
        return first_id, await bus.read(
            "request-1", after_event_id=first_id, block_ms=1
        )

    first_id, replay = asyncio.run(scenario())

    assert first_id == "1000-1"
    assert len(replay) == 1
    assert replay[0].event == "completed"
    assert replay[0].sequence == 2
    assert replay[0].data == {"status": "completed"}
    assert set(redis.expirations.values()) == {3600}


def test_100_concurrent_progress_writes_have_unique_ordered_sequences() -> None:
    redis = FakeStreamRedis()
    bus = RedisEventBus(redis, maxlen=500, ttl_seconds=3600)

    async def scenario():
        await asyncio.gather(
            *(
                bus.publish(
                    request_id="request-concurrent",
                    session_id="session-1",
                    event="trace",
                    data={"node": index},
                )
                for index in range(100)
            )
        )
        return await bus.read(
            "request-concurrent", after_event_id="0-0", block_ms=1, count=200
        )

    events = asyncio.run(scenario())

    assert len(events) == 100
    assert sorted(event.sequence for event in events) == list(range(1, 101))
    assert len({event.event_id for event in events}) == 100


def test_progress_stream_is_capped_instead_of_growing_forever() -> None:
    redis = FakeStreamRedis()
    bus = RedisEventBus(redis, maxlen=500, ttl_seconds=3600)

    async def scenario():
        for index in range(600):
            await bus.publish(
                request_id="request-capped",
                session_id="session-1",
                event="trace",
                data={"node": index},
            )

    asyncio.run(scenario())

    records = redis.streams[bus._stream_key("request-capped")]
    assert len(records) == 500
    assert int(records[0][1]["sequence"]) == 101


class RecordingEventBus:
    def __init__(self, *, fail=False):
        self.events = []
        self.fail = fail

    async def publish(self, **event):
        if self.fail:
            raise ConnectionError("synthetic Redis outage")
        self.events.append(event)
        return f"1000-{len(self.events)}"


class StreamingGraph:
    def __init__(self, state):
        self.state = state
        self.ainvoke_called = False

    async def astream(self, graph_input, *, config, stream_mode):
        assert stream_mode == "updates"
        for node in ["initialize", "single_tool", "finalize_draft"]:
            yield {node: {"private_model_output": "must not be published"}}

    async def aget_state(self, config):
        return SimpleNamespace(values=self.state)

    async def ainvoke(self, graph_input, *, config):
        self.ainvoke_called = True
        return self.state


def test_graph_stream_publishes_only_safe_node_progress_and_uses_checkpoint() -> None:
    state = GraphState(
        request_id="request-1",
        session_id="session-1",
        thread_id="thread-1",
        user_message="Synthetic request",
    ).model_dump(mode="json")
    graph = StreamingGraph(state)
    bus = RecordingEventBus()
    runtime = SimpleNamespace(graph=graph, event_bus=bus)

    result = asyncio.run(
        _invoke_graph_with_progress(
            runtime=runtime,
            graph_input=state,
            config={"configurable": {"thread_id": "thread-1"}},
            request_id="request-1",
            session_id="session-1",
        )
    )

    assert result == state
    assert graph.ainvoke_called is False
    assert [event["data"]["step"] for event in bus.events] == [
        "initialize",
        "single_tool",
        "finalize_draft",
    ]
    assert "private_model_output" not in str(bus.events)


def test_progress_redis_outage_does_not_fail_graph_execution() -> None:
    state = GraphState(
        request_id="request-1",
        session_id="session-1",
        thread_id="thread-1",
        user_message="Synthetic request",
    ).model_dump(mode="json")
    runtime = SimpleNamespace(
        graph=StreamingGraph(state),
        event_bus=RecordingEventBus(fail=True),
    )

    result = asyncio.run(
        _invoke_graph_with_progress(
            runtime=runtime,
            graph_input=state,
            config={"configurable": {"thread_id": "thread-1"}},
            request_id="request-1",
            session_id="session-1",
        )
    )

    assert result == state


class OneBatchBus:
    async def read(self, request_id, *, after_event_id, block_ms, count):
        return [
            ProgressEvent(
                event_id="1000-1",
                event=StreamEventType.TRACE.value,
                sequence=1,
                session_id="session-1",
                request_id=request_id,
                data={"message": "Preparing the draft…"},
            ),
            ProgressEvent(
                event_id="1000-2",
                event=StreamEventType.COMPLETED.value,
                sequence=2,
                session_id="session-1",
                request_id=request_id,
                data={"status": "completed"},
            ),
        ]


class ConnectedRequest:
    async def is_disconnected(self):
        return False


def test_sse_reads_redis_batch_and_stops_on_terminal_event() -> None:
    runtime = SimpleNamespace(event_bus=OneBatchBus(), store=SimpleNamespace())

    async def collect():
        return [
            frame
            async for frame in _redis_event_stream(
                runtime=runtime,
                request=ConnectedRequest(),
                request_id="request-1",
                after_event_id="0-0",
                after_sequence=-1,
            )
        ]

    frames = asyncio.run(collect())

    assert len(frames) == 2
    assert "id: 1000-1" in frames[0]
    assert "event: trace" in frames[0]
    assert "event: completed" in frames[1]


def test_100_idle_sse_connections_do_one_db_status_check_per_redis_wait_cycle() -> None:
    class EmptyBus:
        def __init__(self):
            self.reads = 0

        async def read(self, *args, **kwargs):
            self.reads += 1
            return []

    class CountingStore:
        def __init__(self):
            self.status_checks = 0

        async def get_conversation_run(self, request_id):
            self.status_checks += 1
            return {"request_id": request_id, "status": "running"}

    class OneCycleRequest:
        def __init__(self):
            self.checks = 0

        async def is_disconnected(self):
            self.checks += 1
            return self.checks > 1

    bus = EmptyBus()
    store = CountingStore()
    runtime = SimpleNamespace(event_bus=bus, store=store)

    async def consume_one(index):
        return [
            frame
            async for frame in _redis_event_stream(
                runtime=runtime,
                request=OneCycleRequest(),
                request_id=f"request-{index}",
                after_event_id="0-0",
                after_sequence=-1,
            )
        ]

    async def scenario():
        return await asyncio.gather(*(consume_one(index) for index in range(100)))

    frames = asyncio.run(scenario())

    assert bus.reads == 100
    assert store.status_checks == 100
    assert all(items == [": heartbeat\n\n"] for items in frames)


def test_production_mode_does_not_duplicate_graph_trace_into_postgres() -> None:
    class FailingStore:
        async def list_conversation_events(self, **kwargs):
            raise AssertionError("ordinary progress must not query PostgreSQL")

    state = GraphState(
        request_id="request-1",
        session_id="session-1",
        thread_id="thread-1",
        user_message="Synthetic request",
    )
    runtime = SimpleNamespace(event_bus=RecordingEventBus(), store=FailingStore())

    asyncio.run(_publish_graph_trace(runtime, state))
