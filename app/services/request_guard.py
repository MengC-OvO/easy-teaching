"""Small deterministic safety guard for Main ReAct requests."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple


class RequestGuardAction(str, Enum):
    ALLOW = "allow"
    CLARIFY = "clarify"
    BLOCK = "block"


@dataclass(frozen=True)
class RequestGuardResult:
    action: RequestGuardAction
    code: str
    response: str = ""


_INJECTION_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+"
        r"(?:instructions?|messages?|prompts?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|show|print|repeat|leak)\s+(?:the\s+)?(?:hidden\s+)?"
        r"(?:system|developer)(?:\s+hidden)?\s+(?:prompt|message|instructions?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:jailbreak|developer\s+mode|bypass\s+(?:safety|policy))\b", re.IGNORECASE),
    re.compile(r"(?:忽略|无视|覆盖).{0,12}(?:之前|以上|系统|开发者).{0,8}(?:指令|提示词|消息)"),
    re.compile(r"(?:显示|输出|泄露|复述).{0,10}(?:系统|开发者|隐藏).{0,8}(?:提示词|指令|消息)"),
    re.compile(r"(?:越狱|开发者模式|绕过.{0,6}(?:安全|规则|限制))"),
    re.compile(r"<\s*/?\s*(?:system|developer)\s*>", re.IGNORECASE),
    re.compile(r"\[\s*(?:system|developer)\s*\]", re.IGNORECASE),
)

_HIGH_RISK_PATTERN = re.compile(
    r"(?:diagnos(?:e|is).{0,24}(?:child|student)|prescribe.{0,20}(?:medicine|medication)|"
    r"guarantee.{0,20}(?:legal|compliance)|诊断.{0,12}(?:儿童|孩子|学生)|"
    r"给.{0,8}(?:儿童|孩子).{0,8}(?:开药|处方)|保证.{0,8}(?:合法|合规))",
    re.IGNORECASE,
)

_APPROVAL_BYPASS_PATTERN = re.compile(
    r"(?:pretend|assume).{0,24}approval.{0,24}(?:happened|complete|approved)|"
    r"bypass.{0,16}approval|do\s+not\s+show.{0,20}approval\s+preview|"
    r"directly\s+call\s+save_[a-z_]+.{0,40}invented\s+fields|"
    r"(?:假装|假设).{0,12}(?:审批|批准).{0,12}(?:完成|通过)|"
    r"绕过.{0,8}(?:审批|批准)|不要.{0,12}(?:审批|确认).{0,8}(?:预览|页面)",
    re.IGNORECASE,
)


class EasyTeachingRequestGuard:
    """Block deterministic injection and narrow high-risk requests only.

    Education-scope classification belongs to the local safety model. A keyword
    allowlist here would reject valid teacher language and duplicate that model.
    """

    def evaluate(
        self,
        user_message: str,
        *,
        conversation_context: str = "",
    ) -> RequestGuardResult:
        del conversation_context
        normalized = " ".join(user_message.split())
        if contains_prompt_injection(normalized):
            return RequestGuardResult(
                action=RequestGuardAction.BLOCK,
                code="prompt_injection",
                response=(
                    "I can’t follow instructions to override safety rules, expose internal "
                    "prompts, or bypass tool permissions. I can still help with an Australian "
                    "early-childhood education task."
                ),
            )
        if _HIGH_RISK_PATTERN.search(normalized):
            return RequestGuardResult(
                action=RequestGuardAction.BLOCK,
                code="high_risk_professional_boundary",
                response=(
                    "EasyTeaching can provide general educational and safety information, but it "
                    "cannot diagnose a child, prescribe treatment, or give legal/compliance "
                    "conclusions. Please consult the appropriate qualified professional."
                ),
            )
        if _APPROVAL_BYPASS_PATTERN.search(normalized):
            return RequestGuardResult(
                action=RequestGuardAction.CLARIFY,
                code="approval_bypass_attempt",
                response=(
                    "I cannot pretend approval occurred, invent record fields, or "
                    "bypass the approval preview. Please provide the real observation "
                    "details; any save will still require your review and approval."
                ),
            )
        return RequestGuardResult(
            action=RequestGuardAction.ALLOW,
            code="deterministic_safety_passed",
        )


def contains_prompt_injection(value: str) -> bool:
    return any(pattern.search(value) for pattern in _INJECTION_PATTERNS)


def sanitize_untrusted_prompt_value(value: Any) -> Tuple[Any, int]:
    """Remove instruction-like strings before observations or memory reach the model."""
    if isinstance(value, str):
        if contains_prompt_injection(value):
            return "[removed: suspected prompt-injection instruction]", 1
        return value, 0
    if isinstance(value, dict):
        sanitized = {}
        removed = 0
        for key, nested in value.items():
            safe_nested, nested_removed = sanitize_untrusted_prompt_value(nested)
            sanitized[key] = safe_nested
            removed += nested_removed
        return sanitized, removed
    if isinstance(value, list):
        sanitized_items = []
        removed = 0
        for item in value:
            safe_item, item_removed = sanitize_untrusted_prompt_value(item)
            sanitized_items.append(safe_item)
            removed += item_removed
        return sanitized_items, removed
    return value, 0
