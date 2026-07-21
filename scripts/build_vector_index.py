import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas import KnowledgeChunk
from app.services import (
    ChromaVectorStore,
    GeminiEmbeddingProvider,
    KnowledgeIngestionService,
)
from app.services.model_errors import ModelProviderError


DEFAULT_CHUNKS_PATH = Path("data/knowledge/processed/chunks.jsonl")


def batched(items: List[KnowledgeChunk], batch_size: int) -> Iterable[List[KnowledgeChunk]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_vector_index(
    *,
    chunks_path: Path,
    batch_size: int,
    limit: Optional[int],
    reset: bool,
    resume: bool,
    max_retries: int,
    retry_delay_seconds: float,
    batch_delay_seconds: float,
) -> int:
    ingestion = KnowledgeIngestionService(project_root=ROOT)
    chunks = ingestion.read_chunks_jsonl(chunks_path)
    if limit is not None:
        chunks = chunks[:limit]

    embedding_provider = GeminiEmbeddingProvider()
    vector_store = ChromaVectorStore(reset_collection=reset)

    if resume and not reset:
        existing_ids = vector_store.existing_chunk_ids([chunk.chunk_id for chunk in chunks])
        if existing_ids:
            chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
            print(
                f"resume=true skipped_existing={len(existing_ids)} "
                f"remaining={len(chunks)} collection_count={vector_store.count()}"
            )

    written = 0
    total = len(chunks)
    for batch_number, chunk_batch in enumerate(batched(chunks, batch_size), start=1):
        response = embed_with_retries(
            embedding_provider,
            [chunk.content for chunk in chunk_batch],
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
        )
        written += vector_store.upsert_chunks(chunk_batch, response.vectors)
        print(
            f"batch={batch_number} "
            f"written={written}/{total} "
            f"collection_count={vector_store.count()}"
        )
        if batch_delay_seconds > 0:
            time.sleep(batch_delay_seconds)

    metadata = vector_store.index_metadata()
    print("=== Vector Index ===")
    print(f"collection_name={metadata.collection_name}")
    print(f"index_method={metadata.index_method}")
    print(f"distance_metric={metadata.distance_metric}")
    print(f"embedding_model_name={metadata.embedding_model_name}")
    print(f"embedding_dimension={metadata.embedding_dimension}")
    print(f"index_version={metadata.index_version}")
    print(f"indexed_chunks={vector_store.count()}")
    return written


def embed_with_retries(
    embedding_provider: GeminiEmbeddingProvider,
    texts: List[str],
    *,
    max_retries: int,
    retry_delay_seconds: float,
):
    attempt = 0
    while True:
        try:
            return embedding_provider.embed_texts(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
            )
        except ModelProviderError as error:
            attempt += 1
            retry_after = retry_after_seconds(error)
            print_model_error(
                error,
                attempt=attempt,
                max_retries=max_retries,
                retry_after=retry_after,
            )
            if not error.recoverable or attempt > max_retries:
                raise
            wait_seconds = max(retry_delay_seconds * attempt, retry_after or 0)
            print(f"retry_wait_seconds={wait_seconds:.1f}")
            time.sleep(wait_seconds)


def retry_after_seconds(error: ModelProviderError) -> Optional[float]:
    body = str(error.details.get("body") or "")
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", body, re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def print_model_error(
    error: ModelProviderError,
    *,
    attempt: int,
    max_retries: int,
    retry_after: Optional[float],
) -> None:
    details = error.details
    status_code = details.get("status_code")
    body = details.get("body")
    model = details.get("model")
    print(
        "embedding_error "
        f"attempt={attempt}/{max_retries} "
        f"recoverable={error.recoverable} "
        f"code={error.code.value} "
        f"status_code={status_code} "
        f"model={model} "
        f"provider_retry_after={retry_after}"
    )
    if body:
        print(f"provider_body={body}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the EduFlow Chroma vector index.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the processed chunks JSONL file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of chunks to embed per Gemini request.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of chunks to index for smoke testing.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the configured Chroma collection before indexing.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip chunks that are already present in the collection.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for recoverable embedding provider errors.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=5.0,
        help="Base delay between retries; later retries wait longer.",
    )
    parser.add_argument(
        "--batch-delay-seconds",
        type=float,
        default=0.0,
        help="Delay after every successful batch to avoid provider quota bursts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_vector_index(
        chunks_path=args.chunks,
        batch_size=args.batch_size,
        limit=args.limit,
        reset=args.reset,
        resume=not args.no_resume,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
        batch_delay_seconds=args.batch_delay_seconds,
    )
