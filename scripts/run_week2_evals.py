"""Run the deterministic Week 2 RAG and memory regression set."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import (
    LongTermMemoryCandidate,
    LongTermMemoryScope,
    LongTermMemoryType,
    MemoryRetrievalMode,
    PolicyRAGStatus,
    RerankerMode,
    RetrievalMode,
    ThreadContext,
)
from app.services import ContextManager, EduFlowStore, KnowledgeRetriever, PolicyRAGService
from app.tools import ToolExecutionContext, build_default_tool_registry


DEFAULT_CASES_PATH = ROOT / "data" / "evals" / "week2_cases.json"


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not 15 <= len(cases) <= 20:
        raise ValueError("Week 2 evaluation set must contain 15 to 20 cases.")
    return cases


def evaluate_policy_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    service = PolicyRAGService(
        retriever=KnowledgeRetriever(),
        model_provider=None,
        top_k=3,
        retrieval_mode=RetrievalMode.BM25,
        reranker=RerankerMode.LEXICAL,
    )
    results = []
    for case in cases:
        if case["category"] != "policy_rag":
            continue
        result = service.answer(case["question"])
        actual_sources = sorted({citation.source_id for citation in result.citations})
        expected_status = case["expected_status"]
        expected_source = case.get("expected_source")
        passed = result.status.value == expected_status and (
            expected_source is None or expected_source in actual_sources
        )
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": passed,
                "expected_status": expected_status,
                "actual_status": result.status.value,
                "expected_source": expected_source,
                "actual_sources": actual_sources,
            }
        )
    return results


def _memory_candidate(
    *,
    owner: str,
    content: str,
    retrieval_mode: MemoryRetrievalMode,
) -> LongTermMemoryCandidate:
    return LongTermMemoryCandidate(
        scope=LongTermMemoryScope.TEACHER,
        scope_id=owner,
        memory_type=LongTermMemoryType.TEACHER_PREFERENCE,
        content=content,
        reason="Synthetic Week 2 evaluation fixture.",
        retrieval_mode=retrieval_mode,
        importance=4,
    )


def evaluate_memory_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="eduflow-week2-eval-") as directory:
        store = EduFlowStore(database_url=f"sqlite:///{Path(directory) / 'memory.sqlite3'}")
        store.initialize()
        profile = "Uses Australian English and concise summaries."
        recall = "Previously used water-play activities outdoors."
        store.save_long_term_memory(
            _memory_candidate(
                owner="teacher-001",
                content=profile,
                retrieval_mode=MemoryRetrievalMode.PROFILE,
            )
        )
        store.save_long_term_memory(
            _memory_candidate(
                owner="teacher-001",
                content=recall,
                retrieval_mode=MemoryRetrievalMode.RECALL_ONLY,
            )
        )
        store.save_long_term_memory(
            _memory_candidate(
                owner="teacher-002",
                content="Previously used water-play activities indoors.",
                retrieval_mode=MemoryRetrievalMode.RECALL_ONLY,
            )
        )

        context = ContextManager(long_term_memory_reader=store).build_model_context(
            ThreadContext(), teacher_id="teacher-001"
        )
        registry = build_default_tool_registry(store)
        recalled = registry.execute(
            "recall_long_term_memory",
            {"query": "water-play activities"},
            execution_context=ToolExecutionContext(teacher_id="teacher-001"),
        )
        no_owner = registry.execute(
            "recall_long_term_memory",
            {"query": "water-play"},
            execution_context=ToolExecutionContext(),
        )
        recalled_content = [item["content"] for item in recalled.data["memories"]]
        checks = {
            "profile_is_in_context": profile in context,
            "recall_only_is_not_in_context": recall not in context,
            "matching_recall_is_returned": recalled.success and recall in recalled_content,
            "other_teacher_memory_is_not_returned": all("indoors" not in item for item in recalled_content),
            "missing_owner_is_rejected": not no_owner.success,
        }
    return [
        {
            "id": case["id"],
            "category": case["category"],
            "passed": checks[case["expectation"]],
            "expectation": case["expectation"],
        }
        for case in cases
        if case["category"] == "memory"
    ]


def run(cases_path: Path = DEFAULT_CASES_PATH) -> List[Dict[str, Any]]:
    cases = load_cases(cases_path)
    return [*evaluate_policy_cases(cases), *evaluate_memory_cases(cases)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()
    results = run(args.cases)
    passed = sum(result["passed"] for result in results)
    summary = {"total": len(results), "passed": passed, "failed": len(results) - passed}
    print(json.dumps({"summary": summary, "results": results}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
