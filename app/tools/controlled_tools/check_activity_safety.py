import hashlib
import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
    ToolDomain,
    ToolPermission,
    ToolResult,
)


class CheckActivitySafetyInput(BaseModel):
    activity_text: str = Field(min_length=1)
    age_group: Optional[str] = None
    class_size: Optional[int] = Field(default=None, ge=1)


class SafetyCheckItem(BaseModel):
    code: str
    severity: str
    message: str
    suggestion: str


class CheckActivitySafetyOutput(BaseModel):
    status: str
    issues: List[SafetyCheckItem]
    checked_risk_terms: List[str]
    content_fingerprint: str = Field(min_length=64, max_length=64)
    recovery_content: str = Field(
        min_length=1,
        description=(
            "Exact activity draft inspected by this Tool. Kept out of the normal "
            "compact observation view and used only for failure recovery."
        ),
    )


SAFETY_RISK_RULES: Dict[str, Tuple[str, str, str]] = {
    "water": (
        "high",
        "Water-based activities need explicit supervision, boundaries, and emergency planning.",
        "Add active supervision, safe depth/boundaries, and a dry alternative.",
    ),
    "food": (
        "medium",
        "Food activities need allergy and choking checks.",
        "Check allergies, use age-appropriate textures, and avoid shared utensils.",
    ),
    "allergy": (
        "medium",
        "The draft mentions allergies or possible allergens.",
        "Add allergy checks and substitution options before the activity runs.",
    ),
    "scissors": (
        "medium",
        "Sharp tools need age-appropriate supervision.",
        "Specify child-safe scissors and close educator supervision.",
    ),
    "heat": (
        "high",
        "Heat, cooking, or hot materials need stronger controls.",
        "Keep children away from hot surfaces and use educator-only handling.",
    ),
    "outdoor": (
        "medium",
        "Outdoor activities need weather, boundary, and supervision planning.",
        "Add sun protection, clear boundaries, head counts, and a wet-weather option.",
    ),
}


ADDITIONAL_SAFETY_PATTERNS: Dict[str, Tuple[str, str, str, str]] = {
    "small_loose_parts": (
        r"\b(?:chickpeas?|dried beans?|lentils?|uncooked rice|dry rice|beads?|"
        r"small figurines?|small stones?|pebbles?)\b",
        "high",
        "Small loose materials can create choking, mouthing, or ingestion risks.",
        "Replace them with large non-food pieces that cannot fit through a choke tube, "
        "and document active supervision and pack-away controls.",
    ),
    "scented_materials": (
        r"\b(?:scented|fragranced?|lavender|essential oils?)\b",
        "medium",
        "Scented materials may trigger allergy, asthma, or sensory sensitivities.",
        "Check individual health plans and permissions, avoid essential oils, and offer "
        "an unscented alternative.",
    ),
    "natural_loose_materials": (
        r"\b(?:smooth stones?|pinecones?|pine cones?|twigs?|pieces? of bark|"
        r"textured bark)\b",
        "medium",
        "Natural loose materials need size, condition, toxicity, and mouthing checks.",
        "Use clean non-toxic pieces too large to swallow, remove sharp or splintered "
        "items, and inspect materials before and after use.",
    ),
}


def build_check_activity_safety_tool() -> ToolDefinition:
    def handler(input_data: BaseModel) -> ToolResult:
        data = CheckActivitySafetyInput.model_validate(input_data)
        issues = check_activity_safety(
            data.activity_text,
            age_group=data.age_group,
            class_size=data.class_size,
        )
        status = "needs_revision" if issues else "passed"
        return ToolResult.ok(
            data={
                "status": status,
                "issues": issues,
                "checked_risk_terms": sorted(SAFETY_RISK_RULES),
                "content_fingerprint": activity_content_fingerprint(data.activity_text),
                "recovery_content": data.activity_text,
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="check_activity_safety",
        description=(
            "Check a proposed future activity/learning-experience draft for common "
            "early-childhood safety risks. Do not use for completed observations, "
            "family updates, policy Q&A, saved records, or Drive/export work."
        ),
        category=ToolCategory.SAFETY,
        input_model=CheckActivitySafetyInput,
        output_model=CheckActivitySafetyOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        domain=ToolDomain.INTERNAL,
        parallel_safe=True,
        max_successful_calls_per_run=2,
        max_identical_calls_per_run=1,
        handler=handler,
    )


def activity_content_fingerprint(text: str) -> str:
    """Stable identity for one exact teacher-facing activity version."""

    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_activity_safety(
    activity_text: str,
    *,
    age_group: Optional[str],
    class_size: Optional[int],
) -> List[Dict[str, str]]:
    normalized = activity_text.lower()
    issues: List[Dict[str, str]] = []
    for term, (severity, message, suggestion) in SAFETY_RISK_RULES.items():
        if term in normalized:
            issues.append(
                {
                    "code": f"activity_contains_{term}",
                    "severity": severity,
                    "message": message,
                    "suggestion": suggestion,
                }
            )

    for code, (pattern, severity, message, suggestion) in (
        ADDITIONAL_SAFETY_PATTERNS.items()
    ):
        if re.search(pattern, normalized):
            issues.append(
                {
                    "code": code,
                    "severity": severity,
                    "message": message,
                    "suggestion": suggestion,
                }
            )

    if class_size is not None and class_size >= 20:
        issues.append(
            {
                "code": "large_group_supervision",
                "severity": "medium",
                "message": "Large-group activities need explicit supervision planning.",
                "suggestion": "Split children into small groups or name educator supervision roles.",
            }
        )

    if age_group and re.search(r"\b(0-2|1-2|toddlers?)\b", age_group.lower()):
        issues.append(
            {
                "code": "younger_children_adaptation",
                "severity": "medium",
                "message": "Activities for younger children need choking, mouthing, and mobility adjustments.",
                "suggestion": "Use larger materials, avoid small loose parts, and add close supervision.",
            }
        )

    return issues
