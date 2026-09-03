#!/usr/bin/env python3
"""Evaluate the actual Standard/Deep production retrieval pipelines before gating."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import KnowledgeScope, RetrievalResult  # noqa: E402
from app.services import (  # noqa: E402
    ChatCompletionsModelProvider,
    CrossEncoderReranker,
    KnowledgeIngestionService,
    KnowledgeRetriever,
)
from app.tools.controlled_tools.knowledge_search import (  # noqa: E402
    KnowledgeSearchInput,
    _final_rerank,
    _multi_query_fusion,
    _request,
    _retrieve_async,
    _rewrite_queries_async,
    focus_retrieval_query,
)
from evals.rag_retrieval import (  # noqa: E402
    RagRetrievalCase,
    evaluate_case,
    summarize_mode,
)


async def retrieve(
    *,
    case: RagRetrievalCase,
    mode: str,
    top_k: int,
    retriever: KnowledgeRetriever,
    rewriter: ChatCompletionsModelProvider,
    reranker: CrossEncoderReranker,
) -> RetrievalResult:
    focused = focus_retrieval_query(case.query)
    data = KnowledgeSearchInput(
        query=case.query,
        knowledge_scope=KnowledgeScope(case.scope),
        top_k=min(top_k, 10),
    )
    if mode == "standard":
        return await _retrieve_async(retriever, _request(data, focused, top_k))

    queries = await _rewrite_queries_async(rewriter, focused)
    per_query_top_k = min(20, max(10, top_k * 2))
    results = await asyncio.gather(
        *[
            _retrieve_async(retriever, _request(data, query, per_query_top_k))
            for query in queries
        ]
    )
    candidates = _multi_query_fusion(results, top_k=min(20, max(10, top_k * 4)))
    chunks = await asyncio.to_thread(
        _final_rerank, reranker, focused, candidates, top_k
    )
    return results[0].model_copy(update={"query": focused, "chunks": chunks})


async def evaluate(cases: list[RagRetrievalCase], modes: list[str], ks: list[int]) -> dict:
    retriever = KnowledgeRetriever()
    rewriter = ChatCompletionsModelProvider()
    reranker = await asyncio.to_thread(CrossEncoderReranker)
    chunks = KnowledgeIngestionService(project_root=ROOT).read_chunks_jsonl(
        Path("data/knowledge/processed/chunks.jsonl")
    )
    catalog = {chunk.chunk_id: chunk for chunk in chunks}
    results = []
    total = len(cases) * len(modes)
    completed = 0
    for case in cases:
        for mode in modes:
            completed += 1
            started = perf_counter()
            retrieval = await retrieve(
                case=case,
                mode=mode,
                top_k=max(ks),
                retriever=retriever,
                rewriter=rewriter,
                reranker=reranker,
            )
            measured = evaluate_case(
                case=case,
                mode=mode,
                result=retrieval,
                latency_ms=(perf_counter() - started) * 1000,
                ks=ks,
                chunk_catalog=catalog,
            )
            results.append(measured)
            print(
                f"[{completed:03d}/{total:03d}] {mode:<8} {case.case_id:<44} "
                f"R@3={measured.recall_at_k.get(3, 0):.3f} "
                f"RR={measured.reciprocal_rank:.3f}"
            )

    summaries = {
        mode: summarize_mode(
            [item for item in results if item.mode == mode], ks
        ).model_dump(mode="json")
        for mode in modes
    }
    cases_by_id = {case.case_id: case for case in cases}
    by_scope = {}
    by_tag = {}
    for mode in modes:
        by_scope[mode] = {}
        for scope in sorted({case.scope.value for case in cases}):
            selected = [
                item
                for item in results
                if item.mode == mode
                and cases_by_id[item.case_id].scope.value == scope
            ]
            by_scope[mode][scope] = summarize_mode(selected, ks).model_dump(mode="json")
        by_tag[mode] = {}
        for tag in sorted({tag for case in cases for tag in case.tags}):
            selected = [
                item
                for item in results
                if item.mode == mode and tag in cases_by_id[item.case_id].tags
            ]
            if selected:
                by_tag[mode][tag] = summarize_mode(selected, ks).model_dump(mode="json")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "actual production retrieval before deterministic evidence gating",
        "case_count": len(cases),
        "mode_executions": len(results),
        "ks": ks,
        "query_rewrite_calls": len(cases) if "deep" in modes else 0,
        "summaries": summaries,
        "group_summaries": {"by_scope": by_scope, "by_tag": by_tag},
        "case_results": [item.model_dump(mode="json") for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=["standard", "deep"], default=["standard", "deep"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    args = parser.parse_args()
    raw = []
    for path in args.cases:
        raw.extend(json.loads(path.read_text(encoding="utf-8")))
    cases_by_id = {
        item["case_id"]: RagRetrievalCase.model_validate(item) for item in raw
    }
    cases = list(cases_by_id.values())
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    modes = list(dict.fromkeys(args.modes))
    ks = sorted(set(args.k))
    report = asyncio.run(evaluate(cases, modes, ks))
    report["case_files"] = [str(path) for path in args.cases]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summaries"], indent=2))
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
