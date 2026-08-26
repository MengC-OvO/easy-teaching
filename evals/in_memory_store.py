"""Small in-memory store used only by deterministic offline evals and unit tests."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.schemas.long_memory import (
    LongTermMemoryAction,
    LongTermMemoryCandidate,
    LongTermMemoryOperation,
    MemoryRetrievalMode,
)


SYNTHETIC_CLASS_PROFILES = {
    "kangaroo-room": {
        "class_id": "kangaroo-room",
        "name": "Kangaroo Room",
        "age_group": "3-5",
        "child_count": 18,
        "interests": ["outdoor play", "storytelling", "sensory exploration"],
        "safety_notes": [
            "synthetic data only",
            "check allergies before food play",
        ],
    }
}


class InMemoryEvalStore:
    """Deterministic non-production substitute for database-backed read tools."""

    def __init__(self) -> None:
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._actions: Dict[str, Dict[str, Any]] = {}
        self._records: List[Dict[str, Any]] = [
            {
                "record_type": "observation",
                "class_id": "kangaroo-room",
                "observation_id": "obs-eval-blocks",
                "title": "Block construction observation",
                "observed_at": "2026-08-20T10:00:00",
                "setting": "Kangaroo Room indoor play",
                "objective_text": (
                    "A child stacked six wooden blocks, rebuilt the tower after it "
                    "fell, and invited another child to add a bridge."
                ),
                "status": "draft",
            },
            {
                "record_type": "educational_record",
                "class_id": "kangaroo-room",
                "record_id": "edu-eval-garden",
                "educational_record_type": "learning_story",
                "title": "Garden storytelling",
                "analysis": "Children used natural materials to create and retell a story.",
                "status": "draft",
            },
        ]

    def close(self) -> None:
        self._memories.clear()
        self._actions.clear()


    def get_class_context(
        self,
        *,
        teacher_id: str,
        class_id: str,
        memory_query: Optional[str] = None,
        memory_limit: int = 4,
        include_children: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if teacher_id != "teacher-1":
            raise ValueError("Teacher cannot access the requested class")
        profile = SYNTHETIC_CLASS_PROFILES.get(class_id)
        if profile is None:
            return None
        return {
            "centre_id": "demo-centre",
            "class_id": class_id,
            "name": profile["name"],
            "age_group": profile["age_group"],
            "child_count": profile["child_count"],
            "current_focus": list(profile["interests"]),
            "children": (
                [
                    {
                        "child_id": f"{class_id}-child-{number:02d}",
                        "display_code": f"Child {number:02d}",
                    }
                    for number in range(1, profile["child_count"] + 1)
                ]
                if include_children
                else []
            ),
            "class_memories": [],
        }

    def get_centre_location(self, *, teacher_id: str, class_id: str) -> Dict[str, str]:
        if teacher_id != "teacher-1" or class_id not in SYNTHETIC_CLASS_PROFILES:
            raise ValueError("Teacher cannot access the requested class")
        return {
            "centre_id": "demo-centre",
            "suburb": "Sydney",
            "state": "NSW",
            "timezone": "Australia/Sydney",
            "latitude": -33.8688,
            "longitude": 151.2093,
        }

    def query_records(
        self,
        *,
        teacher_id: str,
        class_id: str,
        record_type: str = "all",
        search_text: Optional[str] = None,
        child_id: Optional[str] = None,
        date_from=None,
        date_to=None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        del child_id, date_from, date_to
        if teacher_id != "teacher-1" or class_id not in SYNTHETIC_CLASS_PROFILES:
            raise ValueError("Teacher cannot access the requested records")
        records = [dict(record) for record in self._records]
        if record_type != "all":
            records = [record for record in records if record["record_type"] == record_type]
        if status is not None:
            records = [record for record in records if record.get("status") == status]
        if search_text:
            needle = search_text.casefold()
            records = [
                record
                for record in records
                if needle
                in " ".join(
                    str(record.get(key) or "")
                    for key in ("title", "analysis", "setting", "objective_text")
                ).casefold()
            ]
        return records[:limit]

    async def create_tool_action_request(
        self,
        *,
        request_id: str,
        session_id: str,
        teacher_id: str,
        class_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        preview: Dict[str, Any],
    ) -> Dict[str, Any]:
        action_id = str(uuid4())
        action = {
            "action_id": action_id,
            "request_id": request_id,
            "session_id": session_id,
            "teacher_id": teacher_id,
            "class_id": class_id,
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "preview": dict(preview),
            "status": "required",
        }
        self._actions[action_id] = action
        return dict(action)

    def save_long_term_memory(
        self,
        candidate: LongTermMemoryCandidate,
    ) -> Dict[str, str]:
        now = datetime.now(timezone.utc).isoformat()
        memory = {
            "memory_id": str(uuid4()),
            "scope": candidate.scope.value,
            "scope_id": candidate.scope_id,
            "memory_type": candidate.memory_type.value,
            "content": candidate.content,
            "reason": candidate.reason,
            "retrieval_mode": candidate.retrieval_mode.value,
            "importance": candidate.importance,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
        }
        self._memories[memory["memory_id"]] = memory
        return self._public_memory(memory)

    def apply_long_term_memory_operation(
        self,
        operation: LongTermMemoryOperation,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> Dict[str, str]:
        if operation.action is LongTermMemoryAction.NOOP:
            return {"action": operation.action.value}
        if operation.action is LongTermMemoryAction.INSERT:
            assert operation.candidate is not None
            candidate = operation.candidate
            self._validate_owner(
                candidate.scope.value,
                candidate.scope_id,
                teacher_id,
                class_id,
            )
            return {
                "action": operation.action.value,
                **self.save_long_term_memory(candidate),
            }

        assert operation.memory_id is not None
        memory = self._memories.get(operation.memory_id)
        if memory is None:
            raise ValueError("Long-term memory does not exist")
        self._validate_owner(
            memory["scope"],
            memory["scope_id"],
            teacher_id,
            class_id,
        )
        if operation.action is LongTermMemoryAction.DELETE:
            memory["is_active"] = False
            memory["updated_at"] = datetime.now(timezone.utc).isoformat()
            return {
                "action": operation.action.value,
                "memory_id": operation.memory_id,
            }

        assert operation.action is LongTermMemoryAction.UPDATE
        assert operation.candidate is not None
        candidate = operation.candidate
        if (
            memory["scope"] != candidate.scope.value
            or memory["scope_id"] != candidate.scope_id
        ):
            raise ValueError("Long-term memory update cannot change its owner")
        memory.update(
            {
                "memory_type": candidate.memory_type.value,
                "content": candidate.content,
                "reason": candidate.reason,
                "retrieval_mode": candidate.retrieval_mode.value,
                "importance": candidate.importance,
                "is_active": True,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"action": operation.action.value, **self._public_memory(memory)}

    def list_long_term_memories(
        self,
        *,
        scope: str,
        scope_id: str,
    ) -> List[Dict[str, str]]:
        return [
            self._public_memory(memory)
            for memory in self._memories.values()
            if memory["scope"] == scope
            and memory["scope_id"] == scope_id
            and memory["is_active"]
        ]

    def list_profile_memories(
        self,
        *,
        teacher_id: Optional[str],
        limit: int = 4,
    ) -> List[Dict[str, str]]:
        if not teacher_id:
            return []
        matches = [
            memory
            for memory in self._memories.values()
            if memory["scope"] == "teacher"
            and memory["scope_id"] == teacher_id
            and memory["memory_type"] == "teacher_preference"
            and memory["retrieval_mode"] == MemoryRetrievalMode.PROFILE.value
            and memory["is_active"]
        ]
        matches.sort(
            key=lambda item: (item["importance"], item["updated_at"]),
            reverse=True,
        )
        return [self._public_memory(item) for item in matches[:limit]]

    def list_memories_for_owners(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        limit: int = 12,
    ) -> List[Dict[str, str]]:
        owners = {("teacher", teacher_id), ("class", class_id)}
        matches = [
            self._public_memory(memory)
            for memory in self._memories.values()
            if (memory["scope"], memory["scope_id"]) in owners
            and memory["is_active"]
        ]
        return matches[-limit:]

    def search_recall_memories(
        self,
        *,
        teacher_id: Optional[str],
        class_id: Optional[str],
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, str]]:
        tokens = [token.lower() for token in query.split() if token]
        return [
            memory
            for memory in self.list_memories_for_owners(
                teacher_id=teacher_id,
                class_id=class_id,
                limit=100,
            )
            if memory["retrieval_mode"] == MemoryRetrievalMode.RECALL_ONLY.value
            and (
                not tokens
                or any(token in memory["content"].lower() for token in tokens)
            )
        ][:limit]

    @staticmethod
    def _validate_owner(
        scope: str,
        scope_id: str,
        teacher_id: Optional[str],
        class_id: Optional[str],
    ) -> None:
        if scope == "teacher" and scope_id != teacher_id:
            raise ValueError("Long-term memory operation is outside the active teacher")
        if scope == "class" and scope_id != class_id:
            raise ValueError("Long-term memory operation is outside the active class")

    @staticmethod
    def _public_memory(memory: Dict[str, Any]) -> Dict[str, str]:
        return {
            "memory_id": memory["memory_id"],
            "scope": memory["scope"],
            "scope_id": memory["scope_id"],
            "memory_type": memory["memory_type"],
            "content": memory["content"],
            "reason": memory["reason"],
            "retrieval_mode": memory["retrieval_mode"],
            "importance": str(memory["importance"]),
            "is_active": str(memory["is_active"]).lower(),
            "created_at": memory["created_at"],
            "updated_at": memory["updated_at"],
        }
