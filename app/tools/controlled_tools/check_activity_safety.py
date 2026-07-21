import re
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.schemas import RiskLevel
from app.tools.definition import (
    ToolCategory,
    ToolDefinition,
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
            },
            risk_level=RiskLevel.L0_READ_ONLY,
        )

    return ToolDefinition(
        name="check_activity_safety",
        description="Check an activity draft for common early childhood safety risks.",
        category=ToolCategory.SAFETY,
        input_model=CheckActivitySafetyInput,
        output_model=CheckActivitySafetyOutput,
        risk_level=RiskLevel.L0_READ_ONLY,
        permission=ToolPermission.AUTO_EXECUTE,
        handler=handler,
    )


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
