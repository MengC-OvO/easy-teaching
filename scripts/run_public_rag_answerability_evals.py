#!/usr/bin/env python3
"""Evaluate answer quality/abstention for public EYLF and NQS evidence only.

Centre-policy cases are deliberately skipped so private/local evidence is never
sent to the configured model provider by this evaluation.
"""

import argparse
import asyncio
import json
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import KnowledgeScope  # noqa: E402
from app.services import (  # noqa: E402
    ChatCompletionsModelProvider,
    ModelMessage,
    ModelRole,
)
from app.tools.controlled_tools.knowledge_search import (  # noqa: E402
    RetrieveKnowledgeInput,
    build_retrieve_knowledge_tool,
)

PUBLIC_SCOPES = {KnowledgeScope.EYLF, KnowledgeScope.NQS}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RagAnswer(StrictModel):
    answerability: Literal["answered", "corrected", "abstained"]
    answer: str = Field(min_length=20)
    cited_evidence_ids: list[str] = Field(default_factory=list, max_length=10)


class RagJudge(StrictModel):
    correctness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    critical_error: bool
    rationale: str = Field(min_length=1, max_length=600)

    @property
    def passed(self) -> bool:
        return not self.critical_error and min(
            self.correctness, self.groundedness, self.completeness
        ) >= 4


ANSWER_SYSTEM = """Use only the supplied public EYLF/NQS evidence and cite supporting
evidence_id values. Use answerability=answered for a supported ordinary answer. If
the evidence directly contradicts a false premise, explain the correction and use
answerability=corrected. If the evidence cannot establish or refute the requested
premise, explicitly say it is not documented/cannot be verified and use
answerability=abstained; do not fill gaps from general knowledge or cite irrelevant
passages."""

JUDGE_SYSTEM = """Judge only against the supplied public evidence. Answerable cases
must be correct, complete and grounded. Correctable false premises must be explicitly
corrected with supporting evidence. Unanswerable cases must abstain without invented
details or irrelevant citation laundering. A fabricated framework outcome, quality
area, element, quote, ratio, diagnosis rule or mandate is a critical error."""


def scopes_for(case: dict) -> list[KnowledgeScope]:
    contracts = case["turns"][0]["expected"]["required_capability_contracts"]
    return list(dict.fromkeys(KnowledgeScope(item["fields"]["knowledge_scope"]) for item in contracts))


def deterministic_checks(case: dict, answer: RagAnswer, judge: RagJudge, valid_ids: set[str]) -> list[dict]:
    expected = case["turns"][0]["expected"]
    target = expected["answerability"]
    lowered = answer.answer.casefold()
    expected_label = {
        "answerable": "answered",
        "correctable": "corrected",
        "unanswerable": "abstained",
    }[target]
    checks = [
        {"name": "answerability", "passed": answer.answerability == expected_label},
        {"name": "citation_ids_valid", "passed": set(answer.cited_evidence_ids) <= valid_ids},
        {"name": "quality_judge", "passed": judge.passed},
        {"name": "citation_policy", "passed": bool(answer.cited_evidence_ids) if target != "unanswerable" else len(answer.cited_evidence_ids) <= 1},
    ]
    # Gold chunk recall is measured separately by run_rag_evals.py. The
    # production tool deliberately exposes turn-local evidence IDs (E1, E2,
    # ...), so comparing those IDs with internal corpus chunk IDs would create
    # a false failure here.
    for terms in expected.get("required_any_terms", []):
        checks.append({"name": "required_any:" + "|".join(terms), "passed": any(term.casefold() in lowered for term in terms)})
    for term in expected.get("required_terms", []):
        checks.append({"name": "required_term:" + term, "passed": term.casefold() in lowered})
    for term in expected.get("forbidden_terms", []):
        checks.append({"name": "forbidden_term:" + term, "passed": term.casefold() not in lowered})
    return checks


def check_rate(items: list[dict], name: str) -> float:
    selected = [
        next(check for check in item["checks"] if check["name"] == name)["passed"]
        for item in items
    ]
    return sum(selected) / len(selected) if selected else 0.0


def latency_percentile(items: list[dict], value: float) -> float:
    ordered = sorted(item["latency_ms"] for item in items)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(value * len(ordered)) - 1)]


async def run(cases_path: Path, modes: list[str]) -> dict:
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    runnable, skipped = [], []
    for case in raw_cases:
        scopes = scopes_for(case)
        if scopes and set(scopes) <= PUBLIC_SCOPES:
            runnable.append((case, scopes))
        else:
            skipped.append({"case_id": case["id"], "reason": "private centre-policy evidence excluded"})

    provider = ChatCompletionsModelProvider()
    judge_provider = ChatCompletionsModelProvider()
    tool = build_retrieve_knowledge_tool()
    results = []
    work = [(case, scopes, mode) for case, scopes in runnable for mode in modes]
    for index, (case, scopes, mode) in enumerate(work, 1):
            started = perf_counter()
            question = case["turns"][0]["message"]
            retrieved = await asyncio.gather(
                *[
                    tool.async_handler(
                        RetrieveKnowledgeInput(
                            query=question,
                            top_k=3 if mode == "standard" else 5,
                            knowledge_scope=scope,
                            mode=mode,
                        )
                    )
                    for scope in scopes
                ]
            )
            evidence = []
            seen = set()
            for retrieval in retrieved:
                for item in retrieval.data["evidence"]:
                    evidence_id = item["evidence_id"]
                    if evidence_id in seen:
                        continue
                    seen.add(evidence_id)
                    citation = item["citation"]
                    evidence.append({
                        "evidence_id": evidence_id,
                        "source_id": citation["source_id"],
                        "title": citation["title"],
                        "section": citation.get("section"),
                        "page": citation.get("page"),
                        "content": item["content"],
                    })
            gate_allowed = all(
                retrieval.data["answerability"] == "answerable"
                for retrieval in retrieved
            )
            if gate_allowed:
                answer_response = await provider.generate_structured(
                    messages=[
                        ModelMessage(role=ModelRole.SYSTEM, content=ANSWER_SYSTEM),
                        ModelMessage(role=ModelRole.USER, content=json.dumps({"question": question, "public_evidence": evidence}, ensure_ascii=False)),
                    ],
                    response_model=RagAnswer,
                    temperature=0.0,
                )
                answer = RagAnswer.model_validate(answer_response.structured)
            else:
                answer = RagAnswer(
                    answerability="abstained",
                    answer=(
                        "The retrieved evidence is not strong enough to answer this "
                        "reliably, so I cannot verify the requested claim."
                    ),
                    cited_evidence_ids=[],
                )
            target = case["turns"][0]["expected"]["answerability"]
            judge_response = await judge_provider.generate_structured(
                messages=[
                    ModelMessage(role=ModelRole.SYSTEM, content=JUDGE_SYSTEM),
                    ModelMessage(role=ModelRole.USER, content=json.dumps({"expected_answerability": target, "question": question, "public_evidence": evidence, "answer": answer.model_dump()}, ensure_ascii=False)),
                ],
                response_model=RagJudge,
                temperature=0.0,
            )
            judge = RagJudge.model_validate(judge_response.structured)
            checks = deterministic_checks(case, answer, judge, seen)
            passed = all(check["passed"] for check in checks)
            results.append({
                "case_id": case["id"],
                "mode": mode,
                "expected_answerability": target,
                "gate_allowed": gate_allowed,
                "answer_model_called": gate_allowed,
                "gate_reasons": [item.data["answerability_reason"] for item in retrieved],
                "passed": passed,
                "latency_ms": (perf_counter() - started) * 1000,
                "retrieved_evidence_ids": sorted(seen),
                "answer": answer.model_dump(),
                "judge": judge.model_dump(),
                "checks": checks,
            })
            print(f"[{index:03d}/{len(work):03d}] {mode:<8} {case['id']:<45} {'PASS' if passed else 'FAIL'} {answer.answerability}")

    summaries = {}
    for mode in modes:
        selected = [item for item in results if item["mode"] == mode]
        answerable = [item for item in selected if item["expected_answerability"] == "answerable"]
        correctable = [item for item in selected if item["expected_answerability"] == "correctable"]
        unanswerable = [item for item in selected if item["expected_answerability"] == "unanswerable"]
        summaries[mode] = {
            "executed": len(selected),
            "skipped_private": len(skipped),
            "passed": sum(item["passed"] for item in selected),
            "pass_rate": sum(item["passed"] for item in selected) / len(selected),
            "answerable_pass_rate": sum(item["passed"] for item in answerable) / len(answerable),
            "correction_pass_rate": (
                sum(item["passed"] for item in correctable) / len(correctable)
                if correctable else None
            ),
            "abstention_pass_rate": sum(item["passed"] for item in unanswerable) / len(unanswerable),
            "answer_model_calls": sum(item["answer_model_called"] for item in selected),
            "answer_model_skips": sum(not item["answer_model_called"] for item in selected),
            "answerability_accuracy": check_rate(selected, "answerability"),
            "citation_id_validity": check_rate(selected, "citation_ids_valid"),
            "citation_policy_pass_rate": check_rate(selected, "citation_policy"),
            "judge_pass_rate": check_rate(selected, "quality_judge"),
            "judge_correctness_mean": sum(item["judge"]["correctness"] for item in selected) / len(selected),
            "judge_groundedness_mean": sum(item["judge"]["groundedness"] for item in selected) / len(selected),
            "judge_completeness_mean": sum(item["judge"]["completeness"] for item in selected) / len(selected),
            "judge_critical_error_rate": sum(item["judge"]["critical_error"] for item in selected) / len(selected),
            "latency_p50_ms": latency_percentile(selected, 0.50),
            "latency_p95_ms": latency_percentile(selected, 0.95),
            "answerability_confusion": {
                target: {
                    prediction: sum(
                        item["expected_answerability"] == target
                        and item["answer"]["answerability"] == prediction
                        for item in selected
                    )
                    for prediction in ["answered", "corrected", "abstained"]
                }
                for target in ["answerable", "correctable", "unanswerable"]
            },
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "production retrieval/evidence gate + real answer model when allowed + independent model judge + deterministic checks",
        "summaries": summaries,
        "results": results,
        "skipped": skipped,
        "limitations": [
            "Agent routing/session behavior was not executed because configured PostgreSQL timed out twice.",
            "Centre-policy answer cases were skipped to prevent private/local evidence disclosure.",
            "The judge uses the same configured model family; deterministic checks are retained separately.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["standard", "deep"],
        default=["standard", "deep"],
    )
    args = parser.parse_args()
    report = asyncio.run(run(args.cases, list(dict.fromkeys(args.modes))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries"], indent=2))
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
