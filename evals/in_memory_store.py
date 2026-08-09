"""Small in-memory store used only by deterministic offline evals and unit tests."""

from datetime import datetime
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

    def close(self) -> None:
        self._memories.clear()

    def get_class_profile(self, class_id: str) -> Optional[Dict[str, Any]]:
        profile = SYNTHETIC_CLASS_PROFILES.get(class_id)
        return None if profile is None else dict(profile)

    def save_long_term_memory(
        self,
        candidate: LongTermMemoryCandidate,
    ) -> Dict[str, str]:
        now = datetime.utcnow().isoformat()
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
            memory["updated_at"] = datetime.utcnow().isoformat()
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
                "updated_at": datetime.utcnow().isoformat(),
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
