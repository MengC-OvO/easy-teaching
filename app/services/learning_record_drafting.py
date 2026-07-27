"""Generate structured learning-record drafts from de-identified observations."""

from typing import Optional, Protocol

from app.schemas import DeidentifiedObservation, LearningRecordDraft
from app.services.model_types import ModelMessage, ModelRequest, ModelResponse, ModelRole
from app.services.observation_redactor import ObservationRedactor


class LearningRecordDraftModel(Protocol):
    """Minimal model boundary used by the documentation drafting service."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...


class LearningRecordDraftingService:
    """Keep raw observations outside the model boundary."""

    def __init__(
        self,
        *,
        model_provider: LearningRecordDraftModel,
        redactor: Optional[ObservationRedactor] = None,
    ) -> None:
        self.model_provider = model_provider
        self.redactor = redactor or ObservationRedactor()

    def create_draft(
        self,
        observation_text: str,
    ) -> tuple[LearningRecordDraft, DeidentifiedObservation]:
        deidentified = self.redactor.deidentify(observation_text)
        return self.create_draft_from_safe_observation(deidentified.safe_text), deidentified

    def create_draft_from_safe_observation(
        self,
        safe_observation: str,
    ) -> LearningRecordDraft:
        """Generate from text that has already crossed the redaction boundary."""
        response = self.model_provider.generate(
            ModelRequest(
                messages=[
                    ModelMessage(role=ModelRole.SYSTEM, content=self._system_prompt()),
                    ModelMessage(
                        role=ModelRole.USER,
                        content=self._user_prompt(safe_observation),
                    ),
                ],
                temperature=0.2,
                response_model=LearningRecordDraft,
            )
        )
        if not isinstance(response.structured, LearningRecordDraft):
            raise ValueError("Model provider did not return a learning-record draft")
        return response.structured

    def revise_draft(
        self,
        observation_text: str,
        revision_instruction: str,
    ) -> tuple[LearningRecordDraft, DeidentifiedObservation, DeidentifiedObservation]:
        """Regenerate a draft without trusting client-supplied draft text."""
        observation = self.redactor.deidentify(observation_text)
        instruction = self.redactor.deidentify(revision_instruction)
        return (
            self.revise_draft_from_safe_text(
                observation.safe_text,
                instruction.safe_text,
            ),
            observation,
            instruction,
        )

    def revise_draft_from_safe_text(
        self,
        safe_observation: str,
        safe_instruction: str,
    ) -> LearningRecordDraft:
        """Regenerate from text that has already crossed the redaction boundary."""
        response = self.model_provider.generate(
            ModelRequest(
                messages=[
                    ModelMessage(role=ModelRole.SYSTEM, content=self._system_prompt()),
                    ModelMessage(
                        role=ModelRole.USER,
                        content=self._revision_prompt(
                            safe_observation,
                            safe_instruction,
                        ),
                    ),
                ],
                temperature=0.2,
                response_model=LearningRecordDraft,
            )
        )
        if not isinstance(response.structured, LearningRecordDraft):
            raise ValueError("Model provider did not return a learning-record draft")
        return response.structured

    def _system_prompt(self) -> str:
        return (
            "You are EduFlow AU, an assistant preparing a teacher-reviewable "
            "early-childhood learning record draft. Return JSON matching the "
            "requested schema. Use only observable facts from the supplied "
            "observation. Do not invent events, diagnoses, medical claims, "
            "family information, or names. Treat tokens such as [CHILD_1] or "
            "[PERSON_NAME_1] as literal privacy placeholders and preserve them. "
            "Keep the analysis tentative and educator-focused."
        )

    def _user_prompt(self, safe_text: str) -> str:
        return (
            "Create a concise learning-record draft from this de-identified "
            "observation:\n\n"
            f"{safe_text}"
        )

    def _revision_prompt(
        self,
        safe_observation: str,
        safe_instruction: str,
    ) -> str:
        return (
            "Regenerate the learning-record draft using the observation and "
            "teacher revision instruction below. Preserve the observable facts "
            "and apply only the requested presentation or emphasis change.\n\n"
            f"De-identified observation:\n{safe_observation}\n\n"
            f"De-identified teacher revision instruction:\n{safe_instruction}"
        )
