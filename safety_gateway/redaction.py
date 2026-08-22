"""High-confidence rules and deterministic placeholder construction."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from safety_gateway.contracts import EntityLabel, InjectionRisk, ModelAnnotation


RESERVED_TOKEN_RE = re.compile(
    r"<(?:PREMASKED_)?(?:PERSON_NAME|PHONE|EMAIL|ADDRESS|DOB)_[1-9][0-9]*>"
)
INTERNAL_TOKEN_RE = re.compile(
    r"(?:<PREMASKED_(?:PERSON_NAME|PHONE|EMAIL|ADDRESS|DOB)_[1-9][0-9]*>"
    r"|<<MODEL:(?:PERSON_NAME|PHONE|EMAIL|ADDRESS|DOB):[1-9][0-9]*>>)"
)


class RedactionError(ValueError):
    """Sanitized deterministic-redaction failure."""


@dataclass(frozen=True)
class PrivateMapping:
    placeholder: str
    value: str
    label: EntityLabel


@dataclass(frozen=True)
class PremaskResult:
    text: str
    mappings: tuple[PrivateMapping, ...]


@dataclass(frozen=True)
class RedactionResult:
    text: str
    mappings: tuple[PrivateMapping, ...]

    @property
    def entity_counts(self) -> dict[EntityLabel, int]:
        return dict(Counter(mapping.label for mapping in self.mappings))


@dataclass(frozen=True)
class RuleSpec:
    label: EntityLabel
    pattern: re.Pattern[str]
    value_group: str | int = 0


RULE_SPECS = (
    RuleSpec(
        EntityLabel.EMAIL,
        re.compile(r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w-])"),
    ),
    RuleSpec(
        EntityLabel.PHONE,
        re.compile(r"(?<!\w)(?:\+?61|0)[2-478](?:[ .()-]*\d){8}(?!\w)"),
    ),
    RuleSpec(
        EntityLabel.DOB,
        re.compile(
            r"(?i)\b(?:DOB|date\s+of\s+birth|born\s+on)\s*(?:is|:|-)?\s*"
            r"(?P<value>(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/](?:19|20)\d{2})"
        ),
        "value",
    ),
    RuleSpec(
        EntityLabel.DOB,
        re.compile(
            r"(?:出生日期|生日)\s*(?:是|为|：|:)?\s*"
            r"(?P<value>(?:19|20)\d{2}[年/-]\d{1,2}[月/-]\d{1,2}日?)"
        ),
        "value",
    ),
)

BLOCK_PATTERNS = (
    re.compile(r"(?i)\b(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+(?:instructions?|messages?|rules?)\b"),
    re.compile(r"(?i)\b(?:bypass|disable|turn\s+off)\b.{0,50}\b(?:privacy|safety|gateway|redaction)\b"),
    re.compile(r"(?i)\b(?:reveal|expose|export|restore)\b.{0,60}\b(?:mapping|system\s+prompt|developer\s+message|private\s+data)\b"),
    re.compile(r"(?:忽略|覆盖).{0,12}(?:系统|开发者|之前).{0,12}(?:指令|提示词)"),
    re.compile(r"(?:绕过|关闭).{0,12}(?:隐私|安全|脱敏|网关)"),
)
SUSPICIOUS_PATTERNS = (
    re.compile(r"(?i)\b(?:system\s+prompt|developer\s+message|hidden\s+instructions?)\b"),
    re.compile(r"(?i)\b(?:decode|execute)\b.{0,30}\bbase64\b"),
    re.compile(r"(?i)\bi\s+g\s+n\s+o\s+r\s+e\b"),
    re.compile(r"(?:系统提示词|开发者消息|隐藏指令)"),
)


def rule_injection_risk(text: str) -> InjectionRisk:
    if any(pattern.search(text) for pattern in BLOCK_PATTERNS):
        return InjectionRisk.BLOCK
    if any(pattern.search(text) for pattern in SUSPICIOUS_PATTERNS):
        return InjectionRisk.SUSPICIOUS
    return InjectionRisk.NORMAL


def _rule_matches(text: str) -> list[tuple[int, int, EntityLabel, str]]:
    candidates: list[tuple[int, int, EntityLabel, str]] = []
    for spec in RULE_SPECS:
        for match in spec.pattern.finditer(text):
            start, end = match.span(spec.value_group)
            candidates.append((start, end, spec.label, match.group(spec.value_group)))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2].value))
    selected: list[tuple[int, int, EntityLabel, str]] = []
    for candidate in candidates:
        if selected and candidate[0] < selected[-1][1]:
            continue
        selected.append(candidate)
    return selected


def premask_text(text: str) -> PremaskResult:
    """Mask only high-confidence values before sending text to Qwen."""
    if RESERVED_TOKEN_RE.search(text) or INTERNAL_TOKEN_RE.search(text):
        raise RedactionError("input contains a reserved safety-gateway token")
    matches = _rule_matches(text)
    counters: Counter[EntityLabel] = Counter()
    mappings: list[PrivateMapping] = []
    output: list[str] = []
    cursor = 0
    for start, end, label, value in matches:
        output.append(text[cursor:start])
        counters[label] += 1
        token = f"<PREMASKED_{label.value}_{counters[label]}>"
        output.append(token)
        mappings.append(PrivateMapping(token, value, label))
        cursor = end
    output.append(text[cursor:])
    return PremaskResult("".join(output), tuple(mappings))


def _model_tokenize(text: str, annotation: ModelAnnotation) -> tuple[str, tuple[PrivateMapping, ...]]:
    matches: list[tuple[int, int, EntityLabel, str]] = []
    for entity in annotation.entities:
        if RESERVED_TOKEN_RE.search(entity.value) or INTERNAL_TOKEN_RE.search(entity.value):
            raise RedactionError("model reported a reserved token")
        start = 0
        found = False
        while True:
            index = text.find(entity.value, start)
            if index < 0:
                break
            found = True
            matches.append((index, index + len(entity.value), entity.label, entity.value))
            start = index + len(entity.value)
        if not found:
            raise RedactionError("model reported a value absent from its input")
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2].value))
    for previous, current in zip(matches, matches[1:]):
        if current[0] < previous[1]:
            raise RedactionError("model entities overlap")

    counters: Counter[EntityLabel] = Counter()
    mappings: list[PrivateMapping] = []
    output: list[str] = []
    cursor = 0
    for start, end, label, value in matches:
        output.append(text[cursor:start])
        counters[label] += 1
        token = f"<<MODEL:{label.value}:{counters[label]}>>"
        output.append(token)
        mappings.append(PrivateMapping(token, value, label))
        cursor = end
    output.append(text[cursor:])
    return "".join(output), tuple(mappings)


def resolve_redaction(original_text: str, premasked: PremaskResult, annotation: ModelAnnotation) -> RedactionResult:
    """Merge rule and model detections, then construct final placeholders once."""
    model_text, model_mappings = _model_tokenize(premasked.text, annotation)
    by_internal_token = {
        mapping.placeholder: mapping for mapping in (*premasked.mappings, *model_mappings)
    }
    counters: Counter[EntityLabel] = Counter()
    final_mappings: list[PrivateMapping] = []
    output: list[str] = []
    cursor = 0
    for match in INTERNAL_TOKEN_RE.finditer(model_text):
        token = match.group(0)
        internal = by_internal_token.get(token)
        if internal is None:
            raise RedactionError("unresolved internal token")
        output.append(model_text[cursor:match.start()])
        counters[internal.label] += 1
        placeholder = f"<{internal.label.value}_{counters[internal.label]}>"
        output.append(placeholder)
        final_mappings.append(PrivateMapping(placeholder, internal.value, internal.label))
        cursor = match.end()
    output.append(model_text[cursor:])
    result = RedactionResult("".join(output), tuple(final_mappings))
    if restore_with_mappings(result.text, result.mappings) != original_text:
        raise RedactionError("deterministic restoration invariant failed")
    return result


def restore_with_mappings(text: str, mappings: tuple[PrivateMapping, ...]) -> str:
    restored = text
    for mapping in mappings:
        restored = restored.replace(mapping.placeholder, mapping.value)
    return restored
