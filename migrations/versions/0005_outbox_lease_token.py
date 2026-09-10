"""Fence expired outbox publishers and executors."""
from alembic import op
import sqlalchemy as sa

revision = "0005_outbox_lease_token"
down_revision = "0004_celery_outbox"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("conversation_task_outbox", sa.Column("lease_token", sa.String(64), nullable=True))


def downgrade():
    op.drop_column("conversation_task_outbox", "lease_token")
