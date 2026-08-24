"""Run one RAG retrieval query without starting the API or full Agent."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import (
    KnowledgeScope,
    RerankerMode,
    RetrievalFilters,
    RetrievalMode,
    RetrievalRequest,
    source_ids_for_scope,
)
from app.services import (
    ChromaVectorStore,
    KnowledgeRetriever,
    LexicalIndexConfigurationError,
    ModelProviderError,
    VectorIndexConfigurationError,
)


DEFAULT_QUERY = "What does the EYLF say about play-based learning?"


def build_scope_filters(scope: KnowledgeScope) -> RetrievalFilters:
    return RetrievalFilters(source_ids=source_ids_for_scope(scope))


def run_retrieval_test(
    *,
    question: str,
    scope: KnowledgeScope,
    mode: RetrievalMode,
    top_k: int,
    reranker: RerankerMode,
) -> int:
    vector_store = ChromaVectorStore() if mode is not RetrievalMode.BM25 else None
    retriever = KnowledgeRetriever(vector_store=vector_store)
    filters = build_scope_filters(scope)
    request = RetrievalRequest(
        query=question,
        top_k=top_k,
        filters=filters,
        mode=mode,
        reranker=reranker,
    )

    started = time.perf_counter()
    result = retriever.retrieve(request)
    latency_ms = (time.perf_counter() - started) * 1000

    print("=== Standalone RAG Retrieval Test ===")
    print(f"question={question}")
    print(f"scope={scope.value}")
    print(f"allowed_source_ids={filters.source_ids or ['all']}")
    print(f"mode={mode.value}")
    print(f"reranker={reranker.value}")
    print(f"api_usage={'none' if mode is RetrievalMode.BM25 else 'one query embedding'}")
    print(f"latency_ms={latency_ms:.2f}")
    print(f"stats={result.stats.model_dump(mode='json')}")
    if vector_store is not None:
        print(f"chroma_collection_count={vector_store.count()}")
    print()

    if not result.chunks:
        print("RESULT_CHECK=FAIL: no evidence returned")
        return 1

    allowed = set(filters.source_ids)
    violations = [
        chunk.citation.source_id
        for chunk in result.chunks
        if allowed and chunk.citation.source_id not in allowed
    ]
    print("=== Retrieved Evidence ===")
    for rank, chunk in enumerate(result.chunks, start=1):
        citation = chunk.citation
        print(f"[{rank}] source_id={citation.source_id}")
        print(f"title={citation.title}")
        print(f"location={format_location(citation.section, citation.page)}")
        print(
            "scores="
            f"dense_distance={format_score(chunk.dense_distance)}, "
            f"bm25={format_score(chunk.bm25_score)}, "
            f"fusion={format_score(chunk.fusion_score)}, "
            f"reranker={format_score(chunk.reranker_score)}"
        )
        print(f"preview={preview(chunk.content)}")
        print()

    if violations:
        print(f"SCOPE_CHECK=FAIL: unexpected_source_ids={sorted(set(violations))}")
        return 2
    print("SCOPE_CHECK=PASS")
    print("RESULT_CHECK=PASS")
    return 0


def format_location(section: Optional[str], page: Optional[int]) -> str:
    parts = []
    if section:
        parts.append(f"section={section}")
    if page:
        parts.append(f"page={page}")
    return ", ".join(parts) if parts else "unknown"


def format_score(score: Optional[float]) -> str:
    return "n/a" if score is None else f"{score:.6f}"


def preview(content: str, *, max_chars: int = 360) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test local RAG retrieval without running the full Agent."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument(
        "--scope",
        choices=[scope.value for scope in KnowledgeScope],
        default=KnowledgeScope.ALL.value,
        help="Hard source boundary: all, eylf, nqs, or centre_policy.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.HYBRID.value,
        help="bm25 is fully local; dense and hybrid use one query embedding.",
    )
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument(
        "--reranker",
        choices=[mode.value for mode in RerankerMode],
        default=RerankerMode.NONE.value,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_retrieval_test(
            question=args.question,
            scope=KnowledgeScope(args.scope),
            mode=RetrievalMode(args.mode),
            top_k=args.top_k,
            reranker=RerankerMode(args.reranker),
        )
    except (
        LexicalIndexConfigurationError,
        ModelProviderError,
        VectorIndexConfigurationError,
    ) as error:
        print("RAG_RETRIEVAL_TEST_FAILED")
        print(str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
