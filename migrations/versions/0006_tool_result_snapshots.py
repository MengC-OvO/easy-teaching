"""Immutable scoped tool result snapshots."""
from alembic import op
import sqlalchemy as sa

revision = "0006_tool_result_snapshots"
down_revision = "0005_outbox_lease_token"
branch_labels = depends_on = None


def upgrade():
    op.create_table("tool_result_snapshots",
        sa.Column("body_ref", sa.String(64), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("teacher_id", sa.String(), nullable=True),
        sa.Column("class_id", sa.String(), nullable=True),
        sa.Column("result_key", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("body", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_tool_result_snapshots_request_id", "tool_result_snapshots", ["request_id"])
    op.create_index("ix_tool_result_snapshots_session_id", "tool_result_snapshots", ["session_id"])


def downgrade():
    op.drop_table("tool_result_snapshots")
