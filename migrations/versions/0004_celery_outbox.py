"""add durable Celery task outbox

Revision ID: 0004_celery_outbox
Revises: 0003_domain_records
"""

from alembic import op
import sqlalchemy as sa


revision = "0004_celery_outbox"
down_revision = "0003_domain_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversation_task_outbox",
        sa.Column(
            "request_id",
            sa.String(),
            sa.ForeignKey("conversation_runs.request_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("publish_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("execution_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("available_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("lease_until", sa.DateTime(), nullable=True),
        sa.Column("celery_task_id", sa.String(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_conversation_task_outbox_status", "conversation_task_outbox", ["status"])
    op.create_index("ix_conversation_task_outbox_available_at", "conversation_task_outbox", ["available_at"])
    op.create_index("ix_conversation_task_outbox_lease_until", "conversation_task_outbox", ["lease_until"])


def downgrade() -> None:
    op.drop_table("conversation_task_outbox")
