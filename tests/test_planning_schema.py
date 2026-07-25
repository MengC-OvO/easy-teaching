import pytest
from pydantic import ValidationError

from app.schemas import (
    ActivityObservationPoint,
    ActivityPlan,
    ActivityPlanEylfAlignment,
    ActivityPlanStep,
    Citation,
)


def valid_activity_plan() -> ActivityPlan:
    return ActivityPlan(
        title="Outdoor texture walk",
        class_profile_summary="Synthetic preschool group interested in outdoor play.",
        learning_goals=[
            "Use descriptive language for natural textures.",
            "Participate safely in shared outdoor exploration.",
        ],
        materials=["Texture cards", "Collection baskets"],
        steps=[
            ActivityPlanStep(
                sequence=1,
                instruction="Review the outdoor boundary and safety expectations.",
                duration_minutes=5,
            ),
            ActivityPlanStep(
                sequence=2,
                instruction="Explore and describe safe natural textures.",
                duration_minutes=15,
            ),
        ],
        observation_points=[
            ActivityObservationPoint(
                focus="Children's descriptive language",
                indicators=[
                    "Uses one or more texture words.",
                    "Shares an observation with a peer.",
                ],
            )
        ],
        eylf_alignments=[
            ActivityPlanEylfAlignment(
                outcome="EYLF Outcome 4",
                rationale="The experience supports curiosity and inquiry.",
                evidence_ids=["E1"],
                citations=[
                    Citation(
                        source="eylf-v2",
                        title="Belonging, Being and Becoming: EYLF V2.0",
                        section="Outcome 4",
                        page=47,
                    )
                ],
            )
        ],
    )


def test_activity_plan_accepts_complete_structured_draft() -> None:
    plan = valid_activity_plan()

    assert plan.title == "Outdoor texture walk"
    assert len(plan.steps) == 2
    assert plan.steps[1].sequence == 2
    assert plan.eylf_alignments[0].citations[0].source == "eylf-v2"
    assert plan.is_draft is True


@pytest.mark.parametrize(
    "field_name",
    [
        "learning_goals",
        "materials",
        "steps",
        "observation_points",
        "eylf_alignments",
    ],
)
def test_activity_plan_rejects_missing_required_sections(field_name: str) -> None:
    data = valid_activity_plan().model_dump()
    data[field_name] = []

    with pytest.raises(ValidationError):
        ActivityPlan.model_validate(data)


def test_activity_plan_rejects_blank_content() -> None:
    data = valid_activity_plan().model_dump()
    data["learning_goals"] = ["   "]

    with pytest.raises(ValidationError):
        ActivityPlan.model_validate(data)


def test_activity_plan_requires_contiguous_ordered_steps() -> None:
    data = valid_activity_plan().model_dump()
    data["steps"][1]["sequence"] = 3

    with pytest.raises(ValidationError, match="contiguous sequence numbers"):
        ActivityPlan.model_validate(data)


def test_activity_plan_cannot_be_marked_as_final_output() -> None:
    data = valid_activity_plan().model_dump()
    data["is_draft"] = False

    with pytest.raises(ValidationError):
        ActivityPlan.model_validate(data)


def test_eylf_alignment_requires_evidence_and_citation() -> None:
    data = valid_activity_plan().model_dump()
    data["eylf_alignments"][0]["evidence_ids"] = []
    data["eylf_alignments"][0]["citations"] = []

    with pytest.raises(ValidationError):
        ActivityPlan.model_validate(data)
