from app.schemas import ConversationMemory, ConversationRole, ConversationTurn
from app.services import ConversationMemoryUpdate, LLMContextSummarizer, ModelResponse


class StubMemoryProvider:
    def __init__(self) -> None:
        self.messages = None
        self.response_model = None

    def generate_structured(self, *, messages, response_model, temperature=0.0):
        self.messages = messages
        self.response_model = response_model
        memory = ConversationMemoryUpdate(
            conversation_goal="Complete an activity draft.",
            important_requirements=["Use simple materials."],
            open_tasks=["Make the plan shorter."],
            compact_summary="An activity draft is in progress.",
        )
        return ModelResponse(content=memory.model_dump_json(), structured=memory)


def test_llm_context_summarizer_returns_structured_memory_update() -> None:
    provider = StubMemoryProvider()
    updater = LLMContextSummarizer(provider)

    memory = updater.update_memory(
        previous_memory=ConversationMemory(),
        current_turns=[
            ConversationTurn(
                role=ConversationRole.USER,
                content="Plan an outdoor activity.",
            )
        ],
        archived_turns=[],
        max_summary_chars=120,
    )

    assert memory.conversation_goal == "Complete an activity draft."
    assert memory.important_requirements == ["Use simple materials."]
    assert provider.response_model is ConversationMemoryUpdate
    assert "Previous structured memory" in provider.messages[1].content
    assert "Plan an outdoor activity." in provider.messages[1].content
