"""Durable approval admission and execution, sharing the conversation Outbox.

Each run has at most one frozen action. Its completed graph task is transitioned
to a new action phase under the Outbox lock; a fresh lease fences old callbacks.
"""
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
import hashlib
import json
from uuid import uuid4

from sqlalchemy import func, select

from app.services.models import (
    ConversationEventRecord, ConversationRunRecord, ConversationRunResultRecord,
    ConversationSessionRecord, ConversationTaskOutboxRecord, ToolActionRequest,
)
from app.services.task_lease import LeaseLostError, execution_lease


_record_transaction = ContextVar("approved_record_transaction", default=None)
RECORD_TOOLS = {"save_observation", "save_educational_record"}


class ApprovedActionStoreMixin:
    @asynccontextmanager
    async def _record_write_session(self):
        borrowed = _record_transaction.get()
        if borrowed is not None and borrowed[0] is self:
            yield borrowed[1]
        else:
            async with self.session_factory() as session:
                yield session

    async def _commit_record_write(self, session):
        borrowed = _record_transaction.get()
        if borrowed is not None and borrowed == (self, session):
            await session.flush()
        else:
            await session.commit()

    async def admit_approved_action(self, *, request_id, session_id, decision):
        """Atomically record consent plus an Outbox task, or reject the action."""
        if decision not in {"approve", "reject"}:
            raise ValueError("Unknown approval decision")
        async with self.session_factory() as session:
            # Same lock order as execution and reconciliation: Outbox -> run -> action.
            task = (await session.execute(select(ConversationTaskOutboxRecord).where(
                ConversationTaskOutboxRecord.request_id == request_id
            ).with_for_update())).scalar_one_or_none()
            run = (await session.execute(select(ConversationRunRecord).where(
                ConversationRunRecord.request_id == request_id
            ).with_for_update())).scalar_one_or_none()
            action = (await session.execute(select(ToolActionRequest).where(
                ToolActionRequest.request_id == request_id
            ).with_for_update())).scalar_one_or_none()
            snapshot = await session.get(ConversationRunResultRecord, request_id)
            conversation = await session.get(ConversationSessionRecord, session_id)
            if (run is None or action is None or snapshot is None or conversation is None
                    or run.session_id != session_id or action.session_id != session_id
                    or snapshot.session_id != session_id
                    or action.teacher_id != conversation.teacher_id
                    or action.class_id != conversation.class_id
                    or snapshot.approval.get("action_id") != action.action_id):
                raise ValueError("Frozen action does not match the trusted conversation")
            if decision == "approve" and action.status in {"queued", "executing", "executed", "unknown", "failed"}:
                return {"status": run.status, "action_id": action.action_id}
            if decision == "reject" and action.status == "rejected":
                return {"status": run.status, "action_id": action.action_id}
            if run.status != "waiting_for_approval" or action.status != "pending":
                raise ValueError("Action is no longer waiting for approval")
            if action.expires_at < datetime.utcnow():
                raise ValueError("Approval has expired")
            self._check_action_hash(action)
            action.decided_at = datetime.utcnow()
            if decision == "reject":
                await self._finish_approved_action(session, action, status="rejected", result={})
            else:
                action.status = "queued"
                run.status = "running"
                snapshot.approval = {**snapshot.approval, "status": "approved", "result": {"execution_status": "queued"}}
                payload = {"kind": "approved_action", "action_id": action.action_id,
                           "session_id": session_id, "thread_id": conversation.thread_id}
                if task is None:
                    task = ConversationTaskOutboxRecord(request_id=request_id, payload=payload)
                    session.add(task)
                task.payload = payload
                task.status = "pending"
                task.lease_token = None
                task.lease_until = None
                task.publish_attempts = 0
                task.execution_attempts = 0
                task.available_at = datetime.utcnow()
                task.updated_at = datetime.utcnow()
                task.last_error = None
                task.celery_task_id = None
                await self._action_event(session, action, "trace", {"step": "approval_queued", "message": "Approved action queued for execution."})
            await session.commit()
            return {"status": run.status, "action_id": action.action_id}

    @staticmethod
    def _check_action_hash(action):
        encoded = json.dumps(action.arguments, ensure_ascii=False, sort_keys=True, default=str)
        if hashlib.sha256(encoded.encode()).hexdigest() != action.arguments_hash:
            raise ValueError("Frozen action failed its integrity check")

    async def _action_event(self, session, action, event, data):
        sequence = (await session.execute(select(func.max(ConversationEventRecord.sequence)).where(
            ConversationEventRecord.request_id == action.request_id
        ))).scalar_one_or_none()
        session.add(ConversationEventRecord(event_id=str(uuid4()), request_id=action.request_id,
            session_id=action.session_id, event=event, sequence=0 if sequence is None else sequence + 1, data=data))

    async def _finish_approved_action(self, session, action, *, status, result):
        run = await session.get(ConversationRunRecord, action.request_id)
        snapshot = await session.get(ConversationRunResultRecord, action.request_id)
        if run is None or snapshot is None:
            raise ValueError("Approved action lost its run/result")
        action.status = status
        action.result = result
        action.executed_at = datetime.utcnow() if status == "executed" else None
        success = status in {"executed", "rejected"}
        run.status = "completed" if success else "failed"
        snapshot.approval = {**snapshot.approval,
            "status": "rejected" if status == "rejected" else "approved" if status == "executed" else "failed",
            "result": result}
        session.add(self._audit_event(teacher_id=action.teacher_id, class_id=action.class_id,
            action=f"approval_{status}", resource_type="tool_action", resource_id=action.action_id,
            tool_name=action.tool_name, result="success" if success else "failed"))
        await self._action_event(session, action, run.status, {"status": run.status,
            "tool_name": action.tool_name, "execution_status": status})

    async def execute_approved_action(self, *, action_id, registry):
        """DB record handlers share the completion transaction; external calls never replay blindly."""
        from app.tools import ToolExecutionContext, ToolPermission
        owner = execution_lease.get()
        if owner is None:
            raise LeaseLostError("Approved action requires a worker lease")
        async with self.session_factory() as session:
            await self._guard_execution_write(session)
            action = (await session.execute(select(ToolActionRequest).where(
                ToolActionRequest.action_id == action_id,
                ToolActionRequest.request_id == owner[0],
            ).with_for_update())).scalar_one_or_none()
            if action is None:
                raise ValueError("Approved action does not belong to this task")
            if action.status in {"executed", "failed", "unknown", "rejected"}:
                return "completed" if action.status in {"executed", "rejected"} else "failed"
            if action.status == "executing":
                await self._finish_approved_action(session, action, status="unknown", result={"error": {
                    "code": "action_outcome_unknown", "message": "Execution was interrupted. Verify the previous result before creating another action; it has not been retried."}})
                await session.commit()
                return "failed"
            if action.status != "queued":
                raise ValueError("Action has not been approved for execution")
            self._check_action_hash(action)
            if action.tool_name.startswith("drive__") and registry.get(action.tool_name) is None:
                await registry.ensure_drive_tools()
            tool = registry.get(action.tool_name)
            if tool is None or tool.permission_for(action.arguments) is not ToolPermission.REQUIRE_APPROVAL:
                raise ValueError("Approved tool definition is missing or no longer requires approval")
            tool.validate_arguments(action.arguments)
            context = ToolExecutionContext(teacher_id=action.teacher_id, class_id=action.class_id,
                session_id=action.session_id, request_id=action.request_id)
            if action.tool_name in RECORD_TOOLS:
                token = _record_transaction.set((self, session))
                try:
                    result = await registry.execute_async(action.tool_name, action.arguments,
                        approved=True, execution_context=context, allowed_tool_names=[action.tool_name])
                    if not result.success:
                        # Roll back any handler writes, including failures after output validation.
                        raise ValueError(result.error.message if result.error else "Record save failed")
                    await self._finish_approved_action(session, action, status="executed", result=result.data)
                    await session.commit()
                    return "completed"
                finally:
                    _record_transaction.reset(token)
            # Commit an intent marker before any filesystem/network side effect.
            action.status = "executing"
            name, arguments = action.tool_name, dict(action.arguments)
            await session.commit()
        result = await registry.execute_async(name, arguments, approved=True,
            execution_context=context, allowed_tool_names=[name])
        async with self.session_factory() as session:
            await self._guard_execution_write(session)
            action = await session.get(ToolActionRequest, action_id)
            await self._finish_approved_action(session, action,
                status="executed" if result.success else "unknown",
                result=result.data if result.success else {"error": {"code": "action_outcome_unknown",
                    "message": "The action did not confirm success. Verify its result before retrying."}})
            await session.commit()
        return "completed" if result.success else "failed"

    async def fail_approved_action(self, *, action_id, message):
        async with self.session_factory() as session:
            await self._guard_execution_write(session)
            action = await session.get(ToolActionRequest, action_id)
            if action is None or action.request_id != execution_lease.get()[0]:
                raise ValueError("Action does not belong to this task")
            if action.status in {"executed", "failed", "unknown", "rejected"}:
                return
            unknown = action.status == "executing"
            await self._finish_approved_action(session, action, status="unknown" if unknown else "failed",
                result={"error": {"code": "action_outcome_unknown" if unknown else "approved_action_failed",
                    "message": "Verify the previous result before retrying." if unknown else message}})
            await session.commit()
