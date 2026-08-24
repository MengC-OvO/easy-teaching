import hashlib
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, List, Optional

from app.schemas import (
    CitationMetadata,
    KnowledgeChunk,
    KnowledgeSourceType,
    RetrievedKnowledgeChunk,
    RetrievalFilters,
)


LEXICAL_INDEX_VERSION = "easyteaching-fts5-v1"


class LexicalIndexConfigurationError(ValueError):
    """Raised when a persisted lexical index is missing or incompatible."""


class SQLiteFTS5KnowledgeIndex:
    """Persistent SQLite FTS5 index using SQLite's built-in BM25 ranking."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise LexicalIndexConfigurationError(
                f"Lexical index not found at {self.path}. Run scripts/build_lexical_index.py."
            )
        self._validate()

    @classmethod
    def build(
        cls,
        path: Path | str,
        chunks: Iterable[KnowledgeChunk],
        *,
        source_digest: str = "",
    ) -> "SQLiteFTS5KnowledgeIndex":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        chunk_list = list(chunks)
        try:
            connection = sqlite3.connect(temporary)
            try:
                cls._create_schema(connection)
                connection.executemany(
                    """
                    INSERT INTO knowledge_fts (
                        search_text, chunk_id, content, content_hash, source_id,
                        source_type, title, version, section, page, uri, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [cls._chunk_row(chunk) for chunk in chunk_list],
                )
                connection.executemany(
                    "INSERT INTO index_manifest (key, value) VALUES (?, ?)",
                    [
                        ("index_version", LEXICAL_INDEX_VERSION),
                        ("chunk_count", str(len(chunk_list))),
                        ("source_digest", source_digest),
                    ],
                )
                connection.commit()
            finally:
                connection.close()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return cls(target)

    @staticmethod
    def digest_file(path: Path | str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters,
    ) -> List[RetrievedKnowledgeChunk]:
        terms = self._query_terms(query)
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms)
        clauses = ["knowledge_fts MATCH ?"]
        parameters: List[object] = [match_query]
        self._add_filter(clauses, parameters, "source_id", filters.source_ids)
        self._add_filter(
            clauses,
            parameters,
            "source_type",
            [source_type.value for source_type in filters.source_types],
        )
        self._add_filter(clauses, parameters, "version", filters.versions)
        parameters.append(top_k)
        sql = f"""
            SELECT chunk_id, content, content_hash, source_id, source_type,
                   title, version, section, page, uri, metadata_json,
                   bm25(knowledge_fts) AS raw_score
            FROM knowledge_fts
            WHERE {' AND '.join(clauses)}
            ORDER BY raw_score ASC
            LIMIT ?
        """
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(sql, parameters).fetchall()

        results: List[RetrievedKnowledgeChunk] = []
        for rank, row in enumerate(rows, start=1):
            raw_score = float(row[11])
            score = max(0.0, -raw_score)
            metadata = json.loads(row[10] or "{}")
            metadata["bm25_score"] = f"{score:.6f}"
            results.append(
                RetrievedKnowledgeChunk(
                    chunk_id=row[0],
                    content=row[1],
                    content_hash=row[2],
                    citation=CitationMetadata(
                        source_id=row[3],
                        source_type=KnowledgeSourceType(row[4]),
                        title=row[5],
                        version=row[6],
                        section=row[7] or None,
                        page=int(row[8]) if row[8] else None,
                        uri=row[9] or None,
                    ),
                    distance=1 / (1 + score),
                    bm25_score=score,
                    bm25_rank=rank,
                    metadata=metadata,
                )
            )
        return results

    def count(self) -> int:
        with closing(sqlite3.connect(self.path)) as connection:
            return int(connection.execute("SELECT count(*) FROM knowledge_fts").fetchone()[0])

    def manifest(self) -> dict[str, str]:
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute("SELECT key, value FROM index_manifest").fetchall()
        return dict(rows)

    def _validate(self) -> None:
        try:
            manifest = self.manifest()
        except sqlite3.DatabaseError as error:
            raise LexicalIndexConfigurationError(
                f"Invalid lexical index at {self.path}: {error}"
            ) from error
        if manifest.get("index_version") != LEXICAL_INDEX_VERSION:
            raise LexicalIndexConfigurationError(
                "Lexical index version does not match this application; rebuild it."
            )

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE VIRTUAL TABLE knowledge_fts USING fts5(
                search_text,
                chunk_id UNINDEXED,
                content UNINDEXED,
                content_hash UNINDEXED,
                source_id UNINDEXED,
                source_type UNINDEXED,
                title UNINDEXED,
                version UNINDEXED,
                section UNINDEXED,
                page UNINDEXED,
                uri UNINDEXED,
                metadata_json UNINDEXED,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        connection.execute(
            "CREATE TABLE index_manifest (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    @staticmethod
    def _chunk_row(chunk: KnowledgeChunk) -> tuple[object, ...]:
        citation = chunk.citation
        return (
            chunk.retrieval_text,
            chunk.chunk_id,
            chunk.content,
            chunk.content_hash,
            citation.source_id,
            citation.source_type.value,
            citation.title,
            citation.version,
            citation.section or "",
            citation.page or 0,
            citation.uri or "",
            json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        return list(dict.fromkeys(re.findall(r"[a-z0-9]+", query.lower())))

    @staticmethod
    def _add_filter(
        clauses: List[str],
        parameters: List[object],
        column: str,
        values: List[str],
    ) -> None:
        if not values:
            return
        placeholders = ", ".join("?" for _ in values)
        clauses.append(f"{column} IN ({placeholders})")
        parameters.extend(values)
