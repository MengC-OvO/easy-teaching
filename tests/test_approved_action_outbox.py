"""Real SQL transactions; PostgreSQL concurrency is exercised separately."""
import asyncio
from datetime import datetime, timedelta
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from app.services.async_store import AsyncEasyTeachingStore
from app.services.models import (
    Base, CentreRecord, ClassRecord, ChildRecord, TeacherRecord, TeacherClassMembershipRecord,
    ConversationSessionRecord, ConversationRunRecord, ConversationRunResultRecord,
    ConversationTaskOutboxRecord, ToolActionRequest, ObservationRecord, EducationalRecord,
)
from app.services.task_lease import bind_execution_lease, LeaseLostError
from app.tools import ToolRegistry, ToolDefinition, ToolCategory, ToolPermission, ToolResult
from app.schemas import RiskLevel, ApprovalSubmitRequest
from app.tools.controlled_tools.records import build_save_observation_tool, build_save_educational_record_tool
from app.tasks import worker
from tests.test_outbox_leases import AsyncSessionShim


class TransactionSession(AsyncSessionShim):
    async def flush(self):
        self.session.flush()


@pytest.fixture
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'approval.sqlite'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(CentreRecord(centre_id="c", name="Centre", suburb="Sydney", state="NSW", timezone="Australia/Sydney"))
        db.add(TeacherRecord(teacher_id="t", centre_id="c", display_name="Teacher"))
        db.add(ClassRecord(class_id="cl", centre_id="c", name="Class", age_group="3-5"))
        db.add(ChildRecord(child_id="child", class_id="cl", display_code="Child 01"))
        db.add(TeacherClassMembershipRecord(membership_id="m", teacher_id="t", class_id="cl"))
        db.add(ConversationSessionRecord(session_id="s", thread_id="thread", teacher_id="t", class_id="cl", status="active"))
        db.add(ConversationRunRecord(request_id="r", session_id="s", status="waiting_for_approval"))
        db.add(ConversationTaskOutboxRecord(request_id="r", payload={"thread_id": "thread"}, status="completed"))
        db.add(ConversationRunResultRecord(request_id="r", session_id="s",
            draft={"content": "Preview", "is_draft": True}, citations=[],
            approval={"status": "required", "action_id": "a", "tool_name": "save_observation"}))
        args = dict(observed_at="2026-09-10T10:00:00", setting="Classroom", objective_text="Child stacked large blocks.",
            child_ids=["child"], educator_actions=None, status="draft", source_request_id=None, idempotency_key="stable-key")
        db.add(ToolActionRequest(action_id="a", request_id="r", session_id="s", teacher_id="t", class_id="cl",
            tool_name="save_observation", arguments=args, arguments_hash=digest(args), preview=args,
            status="pending", expires_at=datetime.utcnow() + timedelta(hours=1)))
        db.commit()
    value = object.__new__(AsyncEasyTeachingStore)
    value.session_factory = lambda: TransactionSession(engine)
    value.test_engine = engine
    yield value
    engine.dispose()


def digest(args):
    return hashlib.sha256(json.dumps(args, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def registry_for(store):
    registry = ToolRegistry()
    registry.register(build_save_observation_tool(store))
    registry.register(build_save_educational_record_tool(store))
    return registry


async def admit(store):
    return await store.admit_approved_action(request_id="r", session_id="s", decision="approve")


async def claim(store):
    await store.claim_conversation_task_for_publish("r", lease_seconds=30)
    return await store.claim_conversation_task_for_execution("r", lease_seconds=60)


def test_consent_and_task_commit_together_and_repeated_approval_is_idempotent(store):
    async def scenario():
        assert (await admit(store))["status"] == "running"
        assert await admit(store) == {"status": "running", "action_id": "a"}
        task = await store.get_conversation_task("r")
        assert task["payload"]["kind"] == "approved_action" and task["status"] == "pending"
        assert (await store.get_tool_action_request("a"))["status"] == "queued"
        with Session(store.test_engine) as db:
            assert not db.scalars(select(ObservationRecord)).all()
    asyncio.run(scenario())


def test_admission_failure_rolls_back_consent_and_outbox(store, monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("injected admission failure")
    monkeypatch.setattr(store, "_action_event", fail)
    with pytest.raises(RuntimeError):
        asyncio.run(admit(store))
    with Session(store.test_engine) as db:
        assert db.get(ToolActionRequest, "a").status == "pending"
        assert db.get(ConversationRunRecord, "r").status == "waiting_for_approval"
        assert db.get(ConversationTaskOutboxRecord, "r").status == "completed"


@pytest.mark.parametrize("educational", [False, True])
def test_record_and_action_result_commit_together_and_redelivery_does_not_write_twice(store, educational):
    if educational:
        with Session(store.test_engine) as db:
            action = db.get(ToolActionRequest, "a")
            action.tool_name = "save_educational_record"
            action.arguments = dict(record_type="program_plan", title="Block activity", analysis="Explore large blocks with the class.", idempotency_key="stable-edu")
            action.arguments_hash = digest(action.arguments)
            db.commit()
    async def scenario():
        await admit(store)
        await store.claim_conversation_task_for_publish("r", lease_seconds=30)
        runtime = SimpleNamespace(store=store, tool_registry=registry_for(store), event_bus=None)
        assert await worker._execute(runtime, "r") == "completed"
        assert await worker._execute(runtime, "r") == "duplicate_or_finished"
        with Session(store.test_engine) as db:
            model = EducationalRecord if educational else ObservationRecord
            assert len(db.scalars(select(model)).all()) == 1
            assert db.get(ToolActionRequest, "a").status == "executed"
            assert db.get(ConversationRunRecord, "r").status == "completed"
            assert db.get(ConversationRunResultRecord, "r").approval["result"]
    asyncio.run(scenario())


def test_crash_after_business_flush_rolls_back_record_then_lease_recovery_succeeds(store, monkeypatch):
    original = store._finish_approved_action
    async def crash(*args, **kwargs):
        raise RuntimeError("crash after record flush")
    async def scenario():
        await admit(store)
        task = await claim(store)
        monkeypatch.setattr(store, "_finish_approved_action", crash)
        with bind_execution_lease("r", task["lease_token"]), pytest.raises(RuntimeError):
            await store.execute_approved_action(action_id="a", registry=registry_for(store))
        with Session(store.test_engine) as db:
            assert not db.scalars(select(ObservationRecord)).all()
            assert db.get(ToolActionRequest, "a").status == "queued"
            db.get(ConversationTaskOutboxRecord, "r").lease_until = datetime.utcnow() - timedelta(seconds=1)
            db.commit()
        assert await store.reconcile_stalled_conversation_tasks(stale_seconds=1200) == 1
        monkeypatch.setattr(store, "_finish_approved_action", original)
        task = await claim(store)
        with bind_execution_lease("r", task["lease_token"]):
            assert await store.execute_approved_action(action_id="a", registry=registry_for(store)) == "completed"
    asyncio.run(scenario())


def test_external_uncertain_result_is_not_reexecuted_after_crash(store):
    calls = []
    class Empty(BaseModel):
        pass
    async def external(data, context):
        calls.append(context.request_id)
        raise asyncio.CancelledError()  # process interrupted after side effect
    registry = ToolRegistry()
    registry.register(ToolDefinition(name="export_records", description="Synthetic export", category=ToolCategory.FILE,
        input_model=Empty, output_model=Empty, permission=ToolPermission.REQUIRE_APPROVAL,
        risk_level=RiskLevel.L2_CONTROLLED_WRITE, async_runtime_handler=external))
    with Session(store.test_engine) as db:
        action = db.get(ToolActionRequest, "a")
        action.tool_name, action.arguments, action.arguments_hash = "export_records", {}, digest({})
        db.commit()
    async def scenario():
        await admit(store)
        task = await claim(store)
        with bind_execution_lease("r", task["lease_token"]), pytest.raises(asyncio.CancelledError):
            await store.execute_approved_action(action_id="a", registry=registry)
        with Session(store.test_engine) as db:
            assert db.get(ToolActionRequest, "a").status == "executing"
            db.get(ConversationTaskOutboxRecord, "r").lease_until = datetime.utcnow() - timedelta(seconds=1)
            db.commit()
        task = await store.claim_conversation_task_for_execution("r", lease_seconds=60)
        with bind_execution_lease("r", task["lease_token"]):
            assert await store.execute_approved_action(action_id="a", registry=registry) == "failed"
        assert calls == ["r"]
        assert (await store.get_tool_action_request("a"))["status"] == "unknown"
    asyncio.run(scenario())


def test_old_graph_lease_cannot_finish_the_new_action_phase(store):
    with Session(store.test_engine) as db:
        task = db.get(ConversationTaskOutboxRecord, "r")
        task.status, task.lease_token = "running", "old-graph-token"
        task.lease_until = datetime.utcnow() + timedelta(seconds=60)
        db.commit()
    async def scenario():
        await admit(store)
        assert not await store.finish_conversation_task_execution("r", lease_token="old-graph-token", status="completed")
        with bind_execution_lease("r", "old-graph-token"), pytest.raises(LeaseLostError):
            await store.execute_approved_action(action_id="a", registry=registry_for(store))
    asyncio.run(scenario())


def test_reject_cannot_reverse_approved_action(store):
    async def scenario():
        await admit(store)
        with pytest.raises(ValueError):
            await store.admit_approved_action(request_id="r", session_id="s", decision="reject")
        assert (await store.get_tool_action_request("a"))["status"] == "queued"
    asyncio.run(scenario())


def test_api_returns_202_without_executing_and_notifies_relay(store, monkeypatch):
    from app.api.routes.approvals import submit_approval
    from app.config import settings
    monkeypatch.setattr(settings, "task_execution_mode", "celery")
    monkeypatch.setattr("app.api.routes.approvals.require_session_owner", lambda *args: None)
    registry = registry_for(store)
    monkeypatch.setattr(registry, "execute_async", AsyncMock(side_effect=AssertionError("API executed tool")))
    runtime = SimpleNamespace(store=store, tool_registry=registry)
    notified = []
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime=runtime,
        outbox_relay=SimpleNamespace(notify=lambda: notified.append(True)))))
    response = asyncio.run(submit_approval("s", ApprovalSubmitRequest(request_id="r", decision="approve"), request, None))
    assert response.status_code == 202
    assert json.loads(response.body)["status"] == "running"
    assert notified == [True]


@pytest.mark.parametrize("invalid", ["expired", "hash", "scope"])
def test_invalid_approval_never_enqueues(store, invalid):
    with Session(store.test_engine) as db:
        action = db.get(ToolActionRequest, "a")
        if invalid == "expired":
            action.expires_at = datetime.utcnow() - timedelta(seconds=1)
        elif invalid == "hash":
            action.arguments_hash = "corrupt"
        else:
            action.teacher_id = "different-teacher"
        db.commit()
    with pytest.raises(ValueError):
        asyncio.run(admit(store))
    with Session(store.test_engine) as db:
        assert db.get(ConversationTaskOutboxRecord, "r").status == "completed"
        assert db.get(ToolActionRequest, "a").status == "pending"


def test_output_validation_failure_rolls_back_business_write(store):
    class InvalidOutput(BaseModel):
        missing_required_field: str
    registry = ToolRegistry()
    tool = build_save_observation_tool(store).model_copy(update={"output_model": InvalidOutput})
    registry.register(tool)
    async def scenario():
        await admit(store)
        task = await claim(store)
        with bind_execution_lease("r", task["lease_token"]), pytest.raises(ValueError):
            await store.execute_approved_action(action_id="a", registry=registry)
        with Session(store.test_engine) as db:
            assert not db.scalars(select(ObservationRecord)).all()
            assert db.get(ToolActionRequest, "a").status == "queued"
    asyncio.run(scenario())


def test_action_retry_exhaustion_preserves_approval_result_without_graph_recovery(store):
    async def scenario():
        await admit(store)
        task = await claim(store)
        runtime = SimpleNamespace(store=store, graph=SimpleNamespace(aget_state=AsyncMock(side_effect=AssertionError("read graph"))))
        await worker._prepare_retry(runtime, "r", RuntimeError("failure"), exhausted=True, lease_token=task["lease_token"])
        assert (await store.get_conversation_run("r"))["status"] == "failed"
        assert (await store.get_tool_action_request("a"))["status"] == "failed"
        assert (await store.get_conversation_task("r"))["status"] == "dead"
    asyncio.run(scenario())


@pytest.mark.parametrize("educational", [False, True])
def test_real_postgres_concurrent_approvals_and_workers_in_isolated_schema(store, educational):
    """Only creates/drops a UUID-named test schema; never clears application tables."""
    import os
    from uuid import uuid4
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from app.asyncio_compat import run_async
    url = os.getenv("APPROVAL_INTEGRATION_DATABASE_URL")
    if not url:
        pytest.skip("Set APPROVAL_INTEGRATION_DATABASE_URL for PostgreSQL concurrency")
    schema = "approval_test_" + uuid4().hex
    model_to_save = EducationalRecord if educational else ObservationRecord
    if educational:
        with Session(store.test_engine) as session:
            session.add(ObservationRecord(observation_id="source", centre_id="c", class_id="cl",
                author_teacher_id="t", observed_at=datetime.utcnow(), setting="Classroom",
                objective_text="Child stacked blocks", status="draft", idempotency_key="source-key"))
            action = session.get(ToolActionRequest, "a")
            action.tool_name = "save_educational_record"
            action.arguments = dict(record_type="learning_story", title="Block exploration",
                analysis="The child explored stacking large blocks.", observation_ids=["source"], idempotency_key="education-key")
            action.arguments_hash = digest(action.arguments)
            session.commit()
    # Copy only the synthetic fixture, not application data.
    seeded = []
    with Session(store.test_engine) as session:
        for table in Base.metadata.sorted_tables:
            model = next(mapper.class_ for mapper in Base.registry.mappers if mapper.local_table is table)
            for row in session.scalars(select(model)).all():
                seeded.append((model, {column.name: getattr(row, column.name) for column in table.columns}))

    async def scenario():
        engine = create_async_engine(url, execution_options={"schema_translate_map": {None: schema}})
        pg = object.__new__(AsyncEasyTeachingStore)
        pg.session_factory = async_sessionmaker(engine, expire_on_commit=False)
        created = False
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
                created = True
                await conn.run_sync(Base.metadata.create_all)
            async with pg.session_factory() as session:
                for model, values in seeded:
                    session.add(model(**values))
                    await session.flush()
                await session.commit()
            admissions = await asyncio.gather(*(admit(pg) for _ in range(10)))
            assert all(item["status"] == "running" for item in admissions)
            # Real PostgreSQL rollback after the business row and FK links flush.
            first = await claim(pg)
            finish = pg._finish_approved_action
            async def crash(*args, **kwargs):
                raise RuntimeError("injected crash before completion commit")
            pg._finish_approved_action = crash
            with bind_execution_lease("r", first["lease_token"]), pytest.raises(RuntimeError):
                await pg.execute_approved_action(action_id="a", registry=registry_for(pg))
            pg._finish_approved_action = finish
            async with pg.session_factory() as session:
                assert not (await session.execute(select(model_to_save))).scalars().all()
                assert (await session.get(ToolActionRequest, "a")).status == "queued"
                task = await session.get(ConversationTaskOutboxRecord, "r")
                task.lease_until = datetime.utcnow() - timedelta(seconds=1)
                await session.commit()
            await pg.reconcile_stalled_conversation_tasks(stale_seconds=1200)
            await pg.claim_conversation_task_for_publish("r", lease_seconds=30)
            runtime = SimpleNamespace(store=pg, tool_registry=registry_for(pg), event_bus=None)
            outcomes = await asyncio.gather(*(worker._execute(runtime, "r") for _ in range(10)))
            assert outcomes.count("completed") == 1
            assert outcomes.count("duplicate_or_finished") == 9
            async with pg.session_factory() as session:
                assert len((await session.execute(select(model_to_save))).scalars().all()) == 1
                assert (await session.get(ToolActionRequest, "a")).status == "executed"
                result = (await session.get(ConversationRunResultRecord, "r")).approval["result"]
                assert result["record_id" if educational else "observation_id"]
        finally:
            if created:
                async with engine.begin() as conn:
                    await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await engine.dispose()
    run_async(scenario())
