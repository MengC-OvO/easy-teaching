import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services.knowledge_ingestion import KnowledgeIngestionService
from app.services.lexical_index import SQLiteFTS5KnowledgeIndex


def build_lexical_index(*, chunks_path: Path, output_path: Path) -> int:
    ingestion = KnowledgeIngestionService(project_root=ROOT)
    chunks = ingestion.read_chunks_jsonl(chunks_path)
    resolved_chunks = chunks_path if chunks_path.is_absolute() else ROOT / chunks_path
    index = SQLiteFTS5KnowledgeIndex.build(
        output_path if output_path.is_absolute() else ROOT / output_path,
        chunks,
        source_digest=SQLiteFTS5KnowledgeIndex.digest_file(resolved_chunks),
    )
    return index.count()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the persistent SQLite FTS5 BM25 index.")
    parser.add_argument("--chunks", type=Path, default=Path("data/knowledge/processed/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(settings.lexical_index_path))
    args = parser.parse_args()
    count = build_lexical_index(chunks_path=args.chunks, output_path=args.output)
    print("LEXICAL_INDEX_BUILD_OK")
    print(f"output={args.output}")
    print(f"indexed_chunks={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
