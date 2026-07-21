import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import KnowledgeIngestionService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest EduFlow AU knowledge sources into processed JSONL chunks."
    )
    parser.add_argument(
        "--manifest",
        default="data/knowledge/sources.json",
        help="Path to the knowledge source manifest.",
    )
    parser.add_argument(
        "--output",
        default="data/knowledge/processed/chunks.jsonl",
        help="Path to write processed chunk JSONL.",
    )
    parser.add_argument(
        "--max-pages-per-pdf",
        type=int,
        default=None,
        help="Optional page limit for faster local testing.",
    )
    args = parser.parse_args()

    service = KnowledgeIngestionService(project_root=PROJECT_ROOT)
    results = service.ingest_all(
        Path(args.manifest),
        max_pages_per_pdf=args.max_pages_per_pdf,
    )
    total_chunks = service.write_chunks_jsonl(results, Path(args.output))

    print("KNOWLEDGE_INGEST_OK")
    print(f"manifest={args.manifest}")
    print(f"output={args.output}")
    print(f"source_count={len(results)}")
    print(f"chunk_count={total_chunks}")
    for result in results:
        print(f"source={result.source_id},chunks={result.chunk_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
