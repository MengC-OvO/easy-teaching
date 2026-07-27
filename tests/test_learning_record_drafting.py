from app.schemas import LearningRecordDraft
from app.services import LearningRecordDraftingService, ModelResponse
import pytest


class RecordingDraftModel:
    def __init__(self) -> None:
        self.request = None

    def generate(self, request) -> ModelResponse:
        self.request = request
        return ModelResponse(
            content="structured draft",
            structured=LearningRecordDraft(
                observation_summary="[PERSON_NAME_1] persisted with balancing blocks.",
                learning_analysis="The observation suggests sustained engagement.",
                possible_next_steps=["Offer varied balancing materials."],
            ),
        )


def test_drafting_service_sends_only_deidentified_text_to_the_model() -> None:
    model = RecordingDraftModel()
    service = LearningRecordDraftingService(model_provider=model)

    draft, deidentified = service.create_draft(
        "Child named Alex Example persisted with balancing blocks."
    )

    assert draft.is_draft is True
    assert deidentified.safe_text == (
        "Child named [PERSON_NAME_1] persisted with balancing blocks."
    )
    assert model.request is not None
    prompt_text = "\n".join(message.content for message in model.request.messages)
    assert "Alex Example" not in prompt_text
    assert "[PERSON_NAME_1]" in prompt_text
    assert model.request.response_model is LearningRecordDraft


def test_drafting_service_rejects_an_unstructured_model_response() -> None:
    class UnstructuredModel:
        def generate(self, request) -> ModelResponse:
            return ModelResponse(content="not valid structured output")

    service = LearningRecordDraftingService(model_provider=UnstructuredModel())

    try:
        service.create_draft("A child explored paint mixing.")
    except ValueError as error:
        assert str(error) == "Model provider did not return a learning-record draft"
    else:
        raise AssertionError("Expected an unstructured model response to fail")


def test_revision_service_sends_only_deidentified_observation_and_instruction() -> None:
    model = RecordingDraftModel()
    service = LearningRecordDraftingService(model_provider=model)

    draft, observation, instruction = service.revise_draft(
        "Child named Alex Example persisted with blocks.",
        "Make it shorter and use alex@example.test for follow-up.",
    )

    assert draft.is_draft is True
    assert observation.safe_text == "Child named [PERSON_NAME_1] persisted with blocks."
    assert instruction.safe_text == (
        "Make it shorter and use [EMAIL_1] for follow-up."
    )
    prompt_text = "\n".join(message.content for message in model.request.messages)
    assert "Alex Example" not in prompt_text
    assert "alex@example.test" not in prompt_text
    assert "[PERSON_NAME_1]" in prompt_text
    assert "[EMAIL_1]" in prompt_text


def test_learning_record_draft_cannot_be_marked_as_approved() -> None:
    with pytest.raises(ValueError, match="is_draft"):
        LearningRecordDraft(
            observation_summary="A child explored paint mixing.",
            learning_analysis="The observation suggests curiosity.",
            possible_next_steps=["Offer additional colour-mixing materials."],
            is_draft=False,
        )
