"""Copy existing EduFlow API and business rows from SQLite to PostgreSQL.

The source file is never modified. Existing target rows are kept, making the
command safe to retry after a partial run.
"""
import argparse
from pathlib import Path
from typing import Dict, List

from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.dialects.postgresql import insert

from app.config import settings


TABLE_ORDER = (
    "class_profiles",
    "conversation_sessions",
    "conversation_runs",
    "conversation_run_results",
    "conversation_approval_decisions",
    "conversation_events",
    "drafts",
    "learning_records",
    "long_term_memories",
)


def migrate(source_path: Path, target_url: str) -> Dict[str, int]:
    if not source_path.exists():
        raise FileNotFoundError(f"SQLite source does not exist: {source_path}")
    if not target_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise ValueError("Target must be a PostgreSQL DATABASE_URL")

    source = create_engine(f"sqlite:///{source_path}", future=True)
    target = create_engine(target_url, future=True)
    source_meta = MetaData()
    target_meta = MetaData()
    source_meta.reflect(bind=source)
    target_meta.reflect(bind=target)
    counts: Dict[str, int] = {}

    try:
        with source.connect() as source_connection, target.begin() as target_connection:
            for table_name in TABLE_ORDER:
                if table_name not in source_meta.tables:
                    counts[table_name] = 0
                    continue
                if table_name not in target_meta.tables:
                    raise RuntimeError(
                        f"Target table {table_name!r} is missing; run Alembic first"
                    )
                rows: List[dict] = [
                    dict(row)
                    for row in source_connection.execute(
                        select(source_meta.tables[table_name])
                    ).mappings()
                ]
                if rows:
                    target_table = target_meta.tables[table_name]
                    primary_key = next(iter(target_table.primary_key.columns))
                    statement = (
                        insert(target_table)
                        .values(rows)
                        .on_conflict_do_nothing()
                        .returning(primary_key)
                    )
                    counts[table_name] = len(target_connection.execute(statement).all())
                else:
                    counts[table_name] = 0
    finally:
        source.dispose()
        target.dispose()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(settings.database_path),
        help="Existing EduFlow SQLite file",
    )
    parser.add_argument(
        "--target-url",
        default=settings.database_url,
        help="PostgreSQL SQLAlchemy URL (defaults to DATABASE_URL)",
    )
    args = parser.parse_args()
    for table, count in migrate(args.source, args.target_url).items():
        print(f"{table}: copied {count}")


if __name__ == "__main__":
    main()
