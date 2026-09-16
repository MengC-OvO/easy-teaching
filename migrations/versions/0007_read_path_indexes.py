"""Add composite indexes for bounded conversation workspace reads."""

from alembic import op


revision = "0007_read_path_indexes"
down_revision = "0006_tool_result_snapshots"
branch_labels = depends_on = None


def upgrade():
    op.create_index(
        "ix_run_results_session_created",
        "conversation_run_results",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_tool_actions_session_status_tool_created",
        "tool_action_requests",
        ["session_id", "status", "tool_name", "created_at"],
    )


def downgrade():
    op.drop_index(
        "ix_tool_actions_session_status_tool_created",
        table_name="tool_action_requests",
    )
    op.drop_index(
        "ix_run_results_session_created",
        table_name="conversation_run_results",
    )
