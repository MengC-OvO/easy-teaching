import argparse
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import (
    KnowledgeSourceType,
    RerankerMode,
    RetrievalFilters,
    RetrievalMode,
    RetrievalRequest,
)
from app.services import ChromaVectorStore, KnowledgeRetriever, ModelProviderError


DEFAULT_QUERY = "What does the EYLF say about play-based learning?"


def query_vector_index(
    *,
    question: str,
    top_k: int,
    source_type: Optional[str],
    source_ids: List[str],
    versions: List[str],
    mode: RetrievalMode,
    reranker: RerankerMode,
) -> int:
    vector_store = ChromaVectorStore()
    retriever = KnowledgeRetriever(vector_store=vector_store)
    filters = build_retrieval_filters(
        source_type=source_type,
        source_ids=source_ids,
        versions=versions,
    )
    result = retriever.retrieve(
        RetrievalRequest(
            query=question,
            top_k=top_k,
            filters=filters,
            mode=mode,
            reranker=reranker,
        )
    )

    metadata = vector_store.index_metadata()
    print("=== Query ===")
    print(question)
    print()
    print("=== Vector Index ===")
    print(f"collection_name={metadata.collection_name}")
    print(f"collection_count={vector_store.count()}")
    print(f"index_method={metadata.index_method}")
    print(f"distance_metric={metadata.distance_metric}")
    print(f"embedding_model_name={metadata.embedding_model_name}")
    print(f"embedding_dimension={metadata.embedding_dimension}")
    print(f"retrieval_mode={result.stats.mode.value}")
    print(f"reranker={result.stats.reranker.value}")
    print(f"stats={result.stats.model_dump(mode='json')}")
    if filters != RetrievalFilters():
        print(f"filters={filters.model_dump(mode='json')}")
    print()
    print("=== Results ===")

    if not result.chunks:
        print("No matching chunks found.")
        return 0

    for index, chunk in enumerate(result.chunks, start=1):
        citation = chunk.citation
        location = citation_location(citation.section, citation.page)
        print(f"[{index}] distance={chunk.distance:.6f}")
        print(f"title={citation.title}")
        print(f"source_id={citation.source_id}")
        print(f"source_type={citation.source_type.value}")
        print(f"version={citation.version}")
        print(f"location={location}")
        if citation.uri:
            print(f"uri={citation.uri}")
        if chunk.metadata:
            print(f"metadata={chunk.metadata}")
        print("content_preview:")
        print(preview(chunk.content))
        print()

    print("VECTOR_QUERY_OK")
    return 0


def build_retrieval_filters(
    *,
    source_type: Optional[str],
    source_ids: List[str],
    versions: List[str],
) -> RetrievalFilters:
    source_types = [KnowledgeSourceType(source_type)] if source_type else []
    return RetrievalFilters(
        source_ids=source_ids,
        source_types=source_types,
        versions=versions,
    )


def citation_location(section: Optional[str], page: Optional[int]) -> str:
    parts = []
    if section:
        parts.append(f"section={section}")
    if page:
        parts.append(f"page={page}")
    return ", ".join(parts) if parts else "unknown"


def preview(content: str, *, max_chars: int = 700) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the EasyTeaching Chroma vector index.")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Question to embed and search for.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to print.",
    )
    parser.add_argument(
        "--where-source-type",
        choices=["official", "synthetic"],
        default=None,
        help="Optional metadata filter for source_type.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Optional source_id filter. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--version",
        action="append",
        default=[],
        help="Optional source version filter. Repeat for multiple versions.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.DENSE.value,
        help="Retrieval strategy to use.",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Legacy alias for --reranker lexical.",
    )
    parser.add_argument(
        "--reranker",
        choices=[mode.value for mode in RerankerMode],
        default=RerankerMode.NONE.value,
        help="Optional reranker to apply after retrieval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return query_vector_index(
            question=args.question,
            top_k=args.top_k,
            source_type=args.where_source_type,
            source_ids=args.source_id,
            versions=args.version,
            mode=RetrievalMode(args.mode),
            reranker=resolve_reranker_mode(args.reranker, args.use_reranker),
        )
    except ModelProviderError as error:
        print("VECTOR_QUERY_FAILED")
        print(error.to_dict())
        return 1


def resolve_reranker_mode(value: str, legacy_use_reranker: bool) -> RerankerMode:
    if legacy_use_reranker and value == RerankerMode.NONE.value:
        return RerankerMode.LEXICAL
    return RerankerMode(value)


if __name__ == "__main__":
    raise SystemExit(main())
