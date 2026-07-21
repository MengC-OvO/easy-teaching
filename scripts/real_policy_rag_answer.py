import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import PolicyRAGStatus, RerankerMode, RetrievalMode
from app.services import (
    ChatCompletionsModelProvider,
    KnowledgeRetriever,
    ModelProviderError,
    PolicyRAGService,
)


DEFAULT_QUESTION = "What does the EYLF say about play-based learning?"


def run_real_policy_rag(
    *,
    question: str,
    top_k: int,
    mode: RetrievalMode,
    reranker: RerankerMode,
) -> int:
    service = PolicyRAGService(
        retriever=KnowledgeRetriever(),
        model_provider=ChatCompletionsModelProvider(),
        top_k=top_k,
        retrieval_mode=mode,
        reranker=reranker,
    )
    result = service.answer(question)

    print("=== Question ===")
    print(question)
    print()

    print("=== Retrieval ===")
    print(json.dumps(result.retrieval.stats.model_dump(mode="json"), indent=2))
    print()

    if result.status is not PolicyRAGStatus.ANSWERED:
        print("=== Status ===")
        print(result.status.value)
        if result.clarification_question:
            print(f"clarification_question={result.clarification_question}")
        if result.refusal_reason:
            print(f"refusal_reason={result.refusal_reason}")
        print()
        print("REAL_POLICY_RAG_DONE")
        return 0

    print("=== Evidence ===")
    for item in result.evidence:
        citation = item.citation
        print(f"[{item.evidence_id}] distance={item.relevance_distance:.6f}")
        print(f"title={citation.title}")
        print(f"source_id={citation.source_id}")
        print(f"source_type={citation.source_type.value}")
        print(f"version={citation.version}")
        print(f"section={citation.section}")
        print(f"page={citation.page}")
        print(f"uri={citation.uri}")
        if item.metadata:
            print(f"metadata={item.metadata}")
        print("content_preview:")
        print(preview(item.content))
        print()

    print("=== LLM Answer ===")
    print(result.answer)
    print()

    print("=== Citations ===")
    for index, citation in enumerate(result.citations, start=1):
        print(
            f"[{index}] {citation.title} | "
            f"source_id={citation.source_id} | "
            f"version={citation.version} | "
            f"section={citation.section} | "
            f"page={citation.page} | "
            f"uri={citation.uri}"
        )

    print()
    print("REAL_POLICY_RAG_OK")
    return 0


def preview(content: str, *, max_chars: int = 500) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real LLM policy RAG and print retrieval evidence plus final answer."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Policy question to answer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of evidence chunks to pass to the answer model.",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RetrievalMode],
        default=RetrievalMode.BM25.value,
        help="Retrieval mode. BM25 avoids Gemini embedding quota.",
    )
    parser.add_argument(
        "--reranker",
        choices=[mode.value for mode in RerankerMode],
        default=RerankerMode.LEXICAL.value,
        help="Reranker mode. lexical avoids Hugging Face network dependency.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_real_policy_rag(
            question=args.question,
            top_k=args.top_k,
            mode=RetrievalMode(args.mode),
            reranker=RerankerMode(args.reranker),
        )
    except ModelProviderError as error:
        print("REAL_POLICY_RAG_FAILED")
        print(json.dumps(error.to_dict(), indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
