"""Create EduFlow operational and business tables.

Revision ID: 0001_initial_postgres
Revises:
"""
from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_initial_postgres"
down_revision: Optional[str] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "class_profiles",
        sa.Column("class_id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("age_group", sa.String(), nullable=False),
        sa.Column("child_count", sa.Integer(), nullable=False),
        sa.Column("interests", sa.JSON(), nullable=False),
        sa.Column("safety_notes", sa.JSON(), nullable=False),
    )
    op.create_table(
        "conversation_sessions",
        sa.Column("session_id", sa.String(), primary_key=True),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("teacher_id", sa.String(), nullable=True),
        sa.Column("class_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_conversation_sessions_thread_id", "conversation_sessions", ["thread_id"], unique=True)
    op.create_index("ix_conversation_sessions_teacher_id", "conversation_sessions", ["teacher_id"])
    op.create_index("ix_conversation_sessions_class_id", "conversation_sessions", ["class_id"])
    op.create_index("ix_conversation_sessions_created_at", "conversation_sessions", ["created_at"])

    op.create_table(
        "conversation_runs",
        sa.Column("request_id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_conversation_runs_session_id", "conversation_runs", ["session_id"])
    op.create_index("ix_conversation_runs_status", "conversation_runs", ["status"])
    op.create_index("ix_conversation_runs_created_at", "conversation_runs", ["created_at"])
    op.create_index(
        "uq_conversation_runs_one_active_per_session",
        "conversation_runs",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('accepted', 'running', 'waiting_for_approval')"),
    )

    op.create_table(
        "conversation_run_results",
        sa.Column("request_id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("draft", sa.JSON(), nullable=False),
        sa.Column("approval", sa.JSON(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_conversation_run_results_session_id", "conversation_run_results", ["session_id"])
    op.create_index("ix_conversation_run_results_created_at", "conversation_run_results", ["created_at"])

    op.create_table(
        "conversation_approval_decisions",
        sa.Column("request_id", sa.String(), primary_key=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_conversation_approval_decisions_session_id", "conversation_approval_decisions", ["session_id"])
    op.create_index("ix_conversation_approval_decisions_created_at", "conversation_approval_decisions", ["created_at"])

    op.create_table(
        "conversation_events",
        sa.Column("event_id", sa.String(), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("event", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("request_id", "sequence", name="uq_conversation_event_request_sequence"),
    )
    op.create_index("ix_conversation_events_request_id", "conversation_events", ["request_id"])
    op.create_index("ix_conversation_events_session_id", "conversation_events", ["session_id"])
    op.create_index("ix_conversation_events_event", "conversation_events", ["event"])
    op.create_index("ix_conversation_events_created_at", "conversation_events", ["created_at"])

    op.create_table(
        "drafts",
        sa.Column("draft_id", sa.String(), primary_key=True),
        sa.Column("idempotency_key", sa.String(), nullable=True, unique=True),
        sa.Column("draft_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
    )
    op.create_table(
        "learning_records",
        sa.Column("record_id", sa.String(), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=False, unique=True),
        sa.Column("teacher_id", sa.String(), nullable=True),
        sa.Column("class_id", sa.String(), nullable=True),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_learning_records_request_id", "learning_records", ["request_id"], unique=True)
    op.create_index("ix_learning_records_teacher_id", "learning_records", ["teacher_id"])
    op.create_index("ix_learning_records_class_id", "learning_records", ["class_id"])
    op.create_index("ix_learning_records_created_at", "learning_records", ["created_at"])

    op.create_table(
        "long_term_memories",
        sa.Column("memory_id", sa.String(), primary_key=True),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("scope_id", sa.String(), nullable=False),
        sa.Column("memory_type", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("retrieval_mode", sa.String(), nullable=False),
        sa.Column("importance", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    for column in ("scope", "scope_id", "memory_type", "retrieval_mode", "importance", "is_active"):
        op.create_index(f"ix_long_term_memories_{column}", "long_term_memories", [column])


def downgrade() -> None:
    for table in (
        "long_term_memories",
        "learning_records",
        "drafts",
        "conversation_events",
        "conversation_approval_decisions",
        "conversation_run_results",
        "conversation_runs",
        "conversation_sessions",
        "class_profiles",
    ):
        op.drop_table(table)
