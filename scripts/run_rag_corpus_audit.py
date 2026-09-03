#!/usr/bin/env python3
"""Audit RAG corpus coverage, integrity, metadata, and duplication."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import ChromaVectorStore, KnowledgeIngestionService  # noqa: E402


def audit_chunks(chunks: list) -> dict:
    source_counts = Counter(chunk.citation.source_id for chunk in chunks)
    type_counts = Counter(chunk.citation.source_type.value for chunk in chunks)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    content_hashes = [chunk.content_hash for chunk in chunks]
    character_lengths = [len(chunk.content.strip()) for chunk in chunks]
    missing = {
        "source_id": sum(not chunk.citation.source_id for chunk in chunks),
        "title": sum(not chunk.citation.title for chunk in chunks),
        "version": sum(not chunk.citation.version for chunk in chunks),
        "uri": sum(not chunk.citation.uri for chunk in chunks),
        "page_and_section": sum(
            chunk.citation.page is None and not chunk.citation.section
            for chunk in chunks
        ),
        "content_hash": sum(not chunk.content_hash for chunk in chunks),
    }
    return {
        "chunk_count": len(chunks),
        "source_count": len(source_counts),
        "chunks_by_source": dict(sorted(source_counts.items())),
        "chunks_by_source_type": dict(sorted(type_counts.items())),
        "duplicate_chunk_id_count": len(chunk_ids) - len(set(chunk_ids)),
        "duplicate_content_hash_count": len(content_hashes) - len(set(content_hashes)),
        "empty_chunk_count": sum(length == 0 for length in character_lengths),
        "short_chunk_count_lt_80_chars": sum(length < 80 for length in character_lengths),
        "long_chunk_count_gt_4000_chars": sum(length > 4000 for length in character_lengths),
        "content_length_chars": {
            "min": min(character_lengths) if character_lengths else 0,
            "mean": sum(character_lengths) / len(character_lengths) if character_lengths else 0,
            "max": max(character_lengths) if character_lengths else 0,
        },
        "missing_metadata": missing,
        "metadata_completeness": (
            1 - sum(missing.values()) / (len(chunks) * len(missing))
            if chunks else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/knowledge/processed/chunks.jsonl"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-vector-store", action="store_true")
    args = parser.parse_args()
    chunks = KnowledgeIngestionService(project_root=ROOT).read_chunks_jsonl(args.chunks)
    metrics = audit_chunks(chunks)
    vector_count = ChromaVectorStore().count() if args.check_vector_store else None
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "vector_store_count": vector_count,
        "vector_store_matches_processed_chunks": (
            vector_count == metrics["chunk_count"] if vector_count is not None else None
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
