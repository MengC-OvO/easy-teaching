import asyncio
from datetime import datetime, UTC
import inspect

from app.services.async_store import AsyncEasyTeachingStore
from app.services.models import ConversationRunRecord, ConversationTaskOutboxRecord


class RecordingSession:
    def __init__(self):
        self.added = []
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def get(self, model, primary_key):
        return None

    def add(self, record):
        self.added.append(record)

    async def commit(self):
        self.commits += 1
        now = datetime.now(UTC)
        for record in self.added:
            if isinstance(record, ConversationRunRecord):
                record.created_at = now
                record.updated_at = now


def test_run_and_outbox_are_added_to_one_session_and_committed_once() -> None:
    session = RecordingSession()
    store = object.__new__(AsyncEasyTeachingStore)
    store.session_factory = lambda: session
    payload = {
        "request_id": "request-1",
        "session_id": "session-1",
        "thread_id": "thread-1",
        "message": "redacted request",
    }

    result = asyncio.run(
        store.create_conversation_run(
            request_id="request-1",
            session_id="session-1",
            task_payload=payload,
        )
    )

    runs = [item for item in session.added if isinstance(item, ConversationRunRecord)]
    outbox = [
        item for item in session.added if isinstance(item, ConversationTaskOutboxRecord)
    ]
    assert len(runs) == 1
    assert len(outbox) == 1
    assert outbox[0].request_id == runs[0].request_id == "request-1"
    assert outbox[0].payload == payload
    assert session.commits == 1
    assert result["created"] is True


def test_outbox_creation_is_not_accidentally_inserted_in_observation_save() -> None:
    source = inspect.getsource(AsyncEasyTeachingStore.save_observation)
    assert "task_payload" not in source
    assert "ConversationTaskOutboxRecord" not in source
