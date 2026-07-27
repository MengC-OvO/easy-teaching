from app.schemas import PIIType
from app.services import ObservationRedactor


def test_redactor_removes_email_phone_and_labelled_person_name() -> None:
    result = ObservationRedactor().deidentify(
        "Child named Alex Example called 0412 345 678. "
        "Please contact alex@example.test afterwards."
    )

    assert result.safe_text == (
        "Child named [PERSON_NAME_1] called [PHONE_1]. "
        "Please contact [EMAIL_1] afterwards."
    )
    assert result.redacted_types == [
        PIIType.EMAIL,
        PIIType.PHONE,
        PIIType.PERSON_NAME,
    ]
    assert result.replacement_count == 3


def test_redactor_preserves_observation_facts_while_replacing_pii() -> None:
    result = ObservationRedactor().deidentify(
        "The child persisted with balancing blocks after the structure fell."
    )

    assert result.safe_text == (
        "The child persisted with balancing blocks after the structure fell."
    )
    assert result.redacted_types == []
    assert result.replacement_count == 0


def test_redactor_uses_distinct_placeholders_for_repeated_sensitive_values() -> None:
    result = ObservationRedactor().deidentify(
        "Contact 0412 345 678 or 0499 888 777."
    )

    assert result.safe_text == "Contact [PHONE_1] or [PHONE_2]."
    assert result.redacted_types == [PIIType.PHONE]
    assert result.replacement_count == 2
