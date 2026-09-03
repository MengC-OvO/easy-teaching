"""Run retrieval-only RAG metrics without invoking the full Agent."""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.schemas import (
    RetrievalFilters,
    RetrievalMode,
    RetrievalRequest,
    RerankerMode,
    source_ids_for_scope,
)
from app.services import (
    ChromaVectorStore,
    CrossEncoderReranker,
    GeminiEmbeddingProvider,
    KnowledgeIngestionService,
    KnowledgeRetriever,
    ModelProviderError,
)
from evals.rag_retrieval import (
    RagCaseMetrics,
    RagRetrievalCase,
    evaluate_case,
    percentile,
    summarize_mode,
)


DEFAULT_CASES_PATH = Path("data/evals/rag_retrieval_cases.json")
DEFAULT_REPORT_PATH = Path("reports/rag_retrieval_report.json")
DEFAULT_CACHE_PATH = Path("data/local/rag_eval_query_embeddings.json")


class CachedQueryEmbeddingProvider:
    """Persist query vectors so repeated metric runs do not spend API quota again."""

    def __init__(self, cache_path: Path) -> None:
        self.provider = GeminiEmbeddingProvider()
        self.cache_path = cache_path
        self.vectors: Dict[str, List[float]] = {}
        self.cache_hits = 0
        self.api_calls = 0
        self.api_latencies_ms: List[float] = []
        self._load()

    def embed_text(self, text: str, *, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
        key = self._key(text, task_type)
        cached = self.vectors.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        started = time.perf_counter()
        vector = self.provider.embed_text(text, task_type=task_type)
        self.api_latencies_ms.append((time.perf_counter() - started) * 1000)
        self.vectors[key] = vector
        self.api_calls += 1
        self._save()
        return vector

    def _key(self, text: str, task_type: str) -> str:
        value = "|".join(
            [
                settings.embedding_model_name,
                str(settings.embedding_dimension),
                task_type,
                " ".join(text.split()),
            ]
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if raw.get("model") != settings.embedding_model_name:
            return
        if raw.get("dimension") != settings.embedding_dimension:
            return
        self.vectors = {
            str(key): [float(value) for value in vector]
            for key, vector in raw.get("vectors", {}).items()
        }

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "model": settings.embedding_model_name,
                    "dimension": settings.embedding_dimension,
                    "vectors": self.vectors,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)


def load_cases(path: Path, split: str) -> List[RagRetrievalCase]:
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    cases = [RagRetrievalCase.model_validate(case) for case in raw_cases]
    return cases if split == "all" else [case for case in cases if case.split == split]


def run_evaluation(
    *,
    cases: Sequence[RagRetrievalCase],
    modes: Sequence[RetrievalMode],
    reranker_mode: RerankerMode,
    ks: Sequence[int],
    cache_path: Path,
) -> tuple[List[RagCaseMetrics], dict]:
    max_k = max(ks)
    needs_dense = any(mode is not RetrievalMode.BM25 for mode in modes)
    embedding_provider = CachedQueryEmbeddingProvider(cache_path) if needs_dense else None
    vector_store = ChromaVectorStore() if needs_dense else None
    reranker = (
        CrossEncoderReranker()
        if reranker_mode is RerankerMode.CROSS_ENCODER
        else None
    )
    retriever = KnowledgeRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        cross_encoder_reranker=reranker,
        candidate_multiplier=(2 if reranker is not None else 4),
    )
    ingestion = KnowledgeIngestionService(project_root=ROOT)
    chunks = ingestion.read_chunks_jsonl(Path("data/knowledge/processed/chunks.jsonl"))
    catalog = {chunk.chunk_id: chunk for chunk in chunks}

    if embedding_provider is not None:
        print("=== Preparing query embeddings (cached between modes and runs) ===")
        for case in cases:
            embedding_provider.embed_text(case.query)

    case_results: List[RagCaseMetrics] = []
    for mode in modes:
        mode_label = (
            mode.value
            if reranker_mode is RerankerMode.NONE
            else f"{mode.value}+{reranker_mode.value}"
        )
        print(f"\n=== Running {mode_label.upper()} ===")
        for index, case in enumerate(cases, start=1):
            request = RetrievalRequest(
                query=case.query,
                top_k=max_k,
                filters=RetrievalFilters(source_ids=source_ids_for_scope(case.scope)),
                mode=mode,
                reranker=reranker_mode,
            )
            started = time.perf_counter()
            result = retriever.retrieve(request)
            latency_ms = (time.perf_counter() - started) * 1000
            measured = evaluate_case(
                case=case,
                mode=mode_label,
                result=result,
                latency_ms=latency_ms,
                ks=ks,
                chunk_catalog=catalog,
            )
            case_results.append(measured)
            print_case_line(index, len(cases), measured, ks)

    usage = {
        "query_embedding_api_calls": embedding_provider.api_calls if embedding_provider else 0,
        "query_embedding_cache_hits": embedding_provider.cache_hits if embedding_provider else 0,
        "query_embedding_api_latency_p50_ms": (
            percentile(embedding_provider.api_latencies_ms, 0.50)
            if embedding_provider and embedding_provider.api_latencies_ms
            else None
        ),
        "query_embedding_api_latency_p95_ms": (
            percentile(embedding_provider.api_latencies_ms, 0.95)
            if embedding_provider and embedding_provider.api_latencies_ms
            else None
        ),
        "chroma_collection_count": vector_store.count() if vector_store else None,
        "retrieval_latency_excludes_query_embedding": True,
    }
    return case_results, usage


def print_case_line(index: int, total: int, result: RagCaseMetrics, ks: Sequence[int]) -> None:
    largest_k = max(ks)
    recall = result.recall_at_k[largest_k]
    status = "FULL" if recall == 1.0 else "PART" if recall > 0 else "MISS"
    first = next((item for item in result.retrieved if item.relevance > 0), None)
    first_rank = first.rank if first else "-"
    print(
        f"[{index:02d}/{total:02d}] {result.case_id:<28} {status:<4} "
        f"R@{largest_k}={result.recall_at_k[largest_k]:.2f} "
        f"first={first_rank} latency={result.latency_ms:.1f}ms"
    )


def print_summary(summaries, ks: Sequence[int]) -> None:
    headers = [
        "mode",
        *[f"R@{k}" for k in ks],
        f"Hit@{max(ks)}",
        f"P@{max(ks)}",
        "MRR",
        "MAP",
        f"nDCG@{max(ks)}",
        "scope_err",
        "cite_ok",
        "ret_P50ms",
        "ret_P95ms",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary.mode,
                *[f"{summary.recall_at_k[k]:.3f}" for k in ks],
                f"{summary.hit_rate_at_k[max(ks)]:.3f}",
                f"{summary.precision_at_k[max(ks)]:.3f}",
                f"{summary.mrr:.3f}",
                f"{summary.map:.3f}",
                f"{summary.ndcg_at_k[max(ks)]:.3f}",
                f"{summary.scope_violation_rate:.3f}",
                f"{summary.citation_correctness:.3f}",
                f"{summary.latency_p50_ms:.1f}",
                f"{summary.latency_p95_ms:.1f}",
            ]
        )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("\n=== RAG Retrieval Metrics ===")
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-only RAG metrics.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in RetrievalMode],
        default=[mode.value for mode in RetrievalMode],
    )
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument(
        "--reranker",
        choices=[mode.value for mode in RerankerMode],
        default=RerankerMode.NONE.value,
        help="Optional final semantic reranker applied after retrieval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ks = sorted(set(args.k))
    if not ks or min(ks) < 1 or max(ks) > 20:
        print("RAG_EVAL_FAILED: --k values must be between 1 and 20")
        return 2
    cases = load_cases(args.cases, args.split)
    if not cases:
        print("RAG_EVAL_FAILED: no cases selected")
        return 2
    modes = [RetrievalMode(mode) for mode in dict.fromkeys(args.modes)]
    reranker_mode = RerankerMode(args.reranker)
    try:
        case_results, usage = run_evaluation(
            cases=cases,
            modes=modes,
            reranker_mode=reranker_mode,
            ks=ks,
            cache_path=args.cache,
        )
    except ModelProviderError as error:
        print("RAG_EVAL_FAILED")
        print(error.to_dict())
        return 1

    mode_labels = list(dict.fromkeys(result.mode for result in case_results))
    summaries = [
        summarize_mode([result for result in case_results if result.mode == label], ks)
        for label in mode_labels
    ]
    print_summary(summaries, ks)
    print(f"\nquery_embedding_api_calls={usage['query_embedding_api_calls']}")
    print(f"query_embedding_cache_hits={usage['query_embedding_cache_hits']}")
    print(
        "query_embedding_api_latency_p50_ms="
        f"{usage['query_embedding_api_latency_p50_ms']}"
    )
    print(
        "query_embedding_api_latency_p95_ms="
        f"{usage['query_embedding_api_latency_p95_ms']}"
    )
    print("retrieval_latency_excludes_query_embedding=true")
    print(f"chroma_collection_count={usage['chroma_collection_count']}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "split": args.split,
        "ks": ks,
        "modes": [mode.value for mode in modes],
        "reranker": reranker_mode.value,
        "usage": usage,
        "summaries": [summary.model_dump(mode="json") for summary in summaries],
        "case_results": [result.model_dump(mode="json") for result in case_results],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report={args.report}")
    print("RAG_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
