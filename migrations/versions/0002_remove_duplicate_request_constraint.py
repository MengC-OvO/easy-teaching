"""Remove the duplicate learning-record request constraint.

Revision ID: 0002_request_unique
Revises: 0001_initial_postgres
"""
from typing import Optional, Sequence, Union

from alembic import op


revision: str = "0002_request_unique"
down_revision: Optional[str] = "0001_initial_postgres"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "learning_records_request_id_key",
        "learning_records",
        type_="unique",
    )


def downgrade() -> None:
    op.create_unique_constraint(
        "learning_records_request_id_key",
        "learning_records",
        ["request_id"],
    )
