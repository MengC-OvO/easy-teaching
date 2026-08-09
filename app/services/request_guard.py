"""Deterministic scope and prompt-injection guard for Main ReAct requests."""

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

_EDUCATION_SCOPE_PATTERN = re.compile(
    r"(?:early\s+childhood|preschool|kindergarten|childcare|daycare|educator|teacher|"
    r"classroom|children?|famil(?:y|ies)|learning\s+stor(?:y|ies)|EYLF|NQS|ACECQA|"
    r"幼儿园|幼教|早教|托儿|教师|老师|班级|儿童|孩子|家长|家庭|教学|课程|学习故事|"
    r"观察记录|活动计划|户外活动|游戏活动|欢迎语)",
    re.IGNORECASE,
)

_OFF_TOPIC_PATTERN = re.compile(
    r"(?:\bstock(?:s|\s+price)?\b|\bshare\s+price\b|\bcrypto(?:currency)?\b|\bbitcoin\b|"
    r"\bforex\b|\btrading\b|\bpython\b|\bjavascript\b|\bprogramming\b|\bsource\s+code\b|"
    r"股票|股价|炒股|加密货币|比特币|外汇交易|编程|程序代码|写代码|赌博|博彩)",
    re.IGNORECASE,
)

_HIGH_RISK_PATTERN = re.compile(
    r"(?:diagnos(?:e|is).{0,24}(?:child|student)|prescribe.{0,20}(?:medicine|medication)|"
    r"guarantee.{0,20}(?:legal|compliance)|诊断.{0,12}(?:儿童|孩子|学生)|"
    r"给.{0,8}(?:儿童|孩子).{0,8}(?:开药|处方)|保证.{0,8}(?:合法|合规))",
    re.IGNORECASE,
)


class EduFlowRequestGuard:
    """Allow education work, clarify ambiguity, and block clear unsafe requests."""

    def evaluate(
        self,
        user_message: str,
        *,
        conversation_context: str = "",
    ) -> RequestGuardResult:
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
                    "EduFlow can provide general educational and safety information, but it "
                    "cannot diagnose a child, prescribe treatment, or give legal/compliance "
                    "conclusions. Please consult the appropriate qualified professional."
                ),
            )
        if _OFF_TOPIC_PATTERN.search(normalized):
            return RequestGuardResult(
                action=RequestGuardAction.BLOCK,
                code="outside_education_scope",
                response=(
                    "EduFlow is limited to Australian early-childhood education work, such as "
                    "activities, EYLF alignment, teacher drafts, class context, and safety "
                    "guidance."
                ),
            )
        if _EDUCATION_SCOPE_PATTERN.search(normalized) or conversation_context.strip():
            return RequestGuardResult(
                action=RequestGuardAction.ALLOW,
                code="education_scope",
            )
        return RequestGuardResult(
            action=RequestGuardAction.CLARIFY,
            code="ambiguous_scope",
            response=(
                "Could you relate this request to your early-childhood teaching, class, "
                "children, families, EYLF, or activity-planning work?"
            ),
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
