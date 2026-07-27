"""Local, deterministic PII redaction for learning-record observations.

This is a defence-in-depth filter, not a claim that arbitrary natural-language
names can be recognised perfectly.  It removes high-confidence patterns before
any future model call.  A production system should also collect a child
reference separately from free text and use an approved, local entity list.
"""

import re
from collections import OrderedDict

from app.schemas.learning_records import DeidentifiedObservation, PIIType


class ObservationRedactor:
    """Replace high-confidence PII with stable placeholders for one request."""

    _EMAIL_PATTERN = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )
    _PHONE_PATTERN = re.compile(
        r"(?<!\w)(?:\+?61[ .-]?)?(?:\(?0?4\)?\d{2}[ .-]?\d{3}[ .-]?\d{3}|"
        r"\(?0?\d{1,2}\)?[ .-]?\d{4}[ .-]?\d{4})(?!\w)"
    )
    _LABELLED_NAME_PATTERN = re.compile(
        r"\b(?:Child|Student|Learner|Name)\s*(?:named|is|:)?\s*"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
    )

    def deidentify(self, observation_text: str) -> DeidentifiedObservation:
        """Return safe text without retaining the supplied observation."""
        counters: OrderedDict[PIIType, int] = OrderedDict()

        def replace(pii_type: PIIType):
            def replacement(_: re.Match[str]) -> str:
                index = counters.setdefault(pii_type, 0) + 1
                counters[pii_type] = index
                return f"[{pii_type.value.upper()}_{index}]"

            return replacement

        safe_text = self._EMAIL_PATTERN.sub(replace(PIIType.EMAIL), observation_text)
        safe_text = self._PHONE_PATTERN.sub(replace(PIIType.PHONE), safe_text)

        def labelled_name_replacement(match: re.Match[str]) -> str:
            index = counters.setdefault(PIIType.PERSON_NAME, 0) + 1
            counters[PIIType.PERSON_NAME] = index
            prefix = match.group(0)[: match.start(1) - match.start(0)]
            return f"{prefix}[PERSON_NAME_{index}]"

        safe_text = self._LABELLED_NAME_PATTERN.sub(
            labelled_name_replacement,
            safe_text,
        )

        return DeidentifiedObservation(
            safe_text=safe_text,
            redacted_types=list(counters.keys()),
            replacement_count=sum(counters.values()),
        )
