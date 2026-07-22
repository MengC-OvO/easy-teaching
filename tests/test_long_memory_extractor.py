from app.schemas import (
    ConversationRole,
    ConversationTurn,
    LongTermMemoryAction,
    LongTermMemoryCandidate,
    LongTermMemoryDecision,
    LongTermMemoryOperation,
    LongTermMemoryScope,
    LongTermMemoryType,
)
from app.services import LLMLongTermMemoryExtractor, ModelResponse


class StubMemoryProvider:
    def __init__(self, decision: LongTermMemoryDecision) -> None:
        self.decision = decision
        self.messages = None
        self.response_model = None

    def generate_structured(self, *, messages, response_model, temperature=0.0):
        self.messages = messages
        self.response_model = response_model
        return ModelResponse(
            content=self.decision.model_dump_json(),
            structured=self.decision,
        )


def _preference_candidate(content: str = "Prefers concise activity-plan steps."):
    return LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id="teacher-001",
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content=content,
        reason="The teacher explicitly stated the preference.",
    )


def test_extractor_returns_an_allowed_insert_decision() -> None:
    provider = StubMemoryProvider(
        LongTermMemoryDecision(
            operations=[
                LongTermMemoryOperation(
                    action=LongTermMemoryAction.INSERT,
                    candidate=_preference_candidate(),
                    reason="A durable teacher preference was explicitly stated.",
                )
            ]
        )
    )
    extractor = LLMLongTermMemoryExtractor(provider)

    operations = extractor.decide(
        teacher_id="teacher-001",
        existing_memories=[],
        turns=[
            ConversationTurn(
                role=ConversationRole.USER,
                content="For future activity plans, keep the steps concise.",
            )
        ],
    )

    assert len(operations) == 1
    assert operations[0].action is LongTermMemoryAction.INSERT
    assert "teacher-001" in provider.messages[1].content


def test_extractor_passes_existing_memories_and_allows_update_of_owned_record() -> None:
    provider = StubMemoryProvider(
        LongTermMemoryDecision(
            operations=[
                LongTermMemoryOperation(
                    action=LongTermMemoryAction.UPDATE,
                    memory_id="memory-001",
                    candidate=_preference_candidate("Prefers detailed activity-plan steps."),
                    reason="The teacher explicitly changed the preference.",
                )
            ]
        )
    )
    extractor = LLMLongTermMemoryExtractor(provider)
    existing = [
        {
            "memory_id": "memory-001",
            "scope": "teacher",
            "scope_id": "teacher-001",
            "memory_type": "teacher_preference",
            "content": "Prefers concise activity-plan steps.",
            "reason": "Earlier feedback.",
        }
    ]

    operations = extractor.decide(
        teacher_id="teacher-001",
        existing_memories=existing,
        turns=[ConversationTurn(role=ConversationRole.USER, content="Make them detailed now.")],
    )

    assert [operation.action for operation in operations] == [LongTermMemoryAction.UPDATE]
    assert "memory-001" in provider.messages[1].content


def test_extractor_discards_updates_for_memories_outside_the_active_owner() -> None:
    provider = StubMemoryProvider(
        LongTermMemoryDecision(
            operations=[
                LongTermMemoryOperation(
                    action=LongTermMemoryAction.UPDATE,
                    memory_id="memory-002",
                    candidate=_preference_candidate(),
                    reason="Attempted update.",
                )
            ]
        )
    )
    extractor = LLMLongTermMemoryExtractor(provider)

    operations = extractor.decide(
        teacher_id="teacher-001",
        existing_memories=[
            {
                "memory_id": "memory-002",
                "scope": "teacher",
                "scope_id": "teacher-002",
                "memory_type": "teacher_preference",
                "content": "Other teacher preference.",
                "reason": "Earlier feedback.",
            }
        ],
        turns=[ConversationTurn(role=ConversationRole.USER, content="Keep it concise.")],
    )

    assert operations == []


def test_extractor_skips_the_model_call_without_an_owner() -> None:
    provider = StubMemoryProvider(LongTermMemoryDecision())
    extractor = LLMLongTermMemoryExtractor(provider)

    operations = extractor.decide(
        existing_memories=[],
        turns=[ConversationTurn(role=ConversationRole.USER, content="Keep it concise.")],
    )

    assert operations == []
    assert provider.messages is None
