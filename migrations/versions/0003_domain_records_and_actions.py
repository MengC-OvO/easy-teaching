"""Add centre-scoped records, approvals, audit, and knowledge source metadata.

Revision ID: 0003_domain_records
Revises: 0002_request_unique
"""

from typing import Optional, Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_domain_records"
down_revision: Optional[str] = "0002_request_unique"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "centres",
        sa.Column("centre_id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("suburb", sa.String(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_centres_active", "centres", ["active"])

    op.create_table(
        "teachers",
        sa.Column("teacher_id", sa.String(), primary_key=True),
        sa.Column("auth_user_id", sa.String(), nullable=True, unique=True),
        sa.Column("centre_id", sa.String(), sa.ForeignKey("centres.centre_id"), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    for column in ("auth_user_id", "centre_id", "active"):
        op.create_index(f"ix_teachers_{column}", "teachers", [column])

    op.create_table(
        "classes",
        sa.Column("class_id", sa.String(), primary_key=True),
        sa.Column("centre_id", sa.String(), sa.ForeignKey("centres.centre_id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("age_group", sa.String(), nullable=False),
        sa.Column("current_focus", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_classes_centre_id", "classes", ["centre_id"])
    op.create_index("ix_classes_active", "classes", ["active"])

    op.create_table(
        "teacher_class_memberships",
        sa.Column("membership_id", sa.String(), primary_key=True),
        sa.Column("teacher_id", sa.String(), sa.ForeignKey("teachers.teacher_id"), nullable=False),
        sa.Column("class_id", sa.String(), sa.ForeignKey("classes.class_id"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("teacher_id", "class_id", name="uq_teacher_class_membership"),
    )
    for column in ("teacher_id", "class_id", "active"):
        op.create_index(f"ix_teacher_class_memberships_{column}", "teacher_class_memberships", [column])

    op.create_table(
        "children",
        sa.Column("child_id", sa.String(), primary_key=True),
        sa.Column("class_id", sa.String(), sa.ForeignKey("classes.class_id"), nullable=False),
        sa.Column("display_code", sa.String(), nullable=False),
        sa.Column("preferred_name_encrypted", sa.Text(), nullable=True),
        sa.Column("date_of_birth_encrypted", sa.Text(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    for column in ("class_id", "display_code", "active"):
        op.create_index(f"ix_children_{column}", "children", [column])

    op.create_table(
        "observations",
        sa.Column("observation_id", sa.String(), primary_key=True),
        sa.Column("centre_id", sa.String(), sa.ForeignKey("centres.centre_id"), nullable=False),
        sa.Column("class_id", sa.String(), sa.ForeignKey("classes.class_id"), nullable=False),
        sa.Column("author_teacher_id", sa.String(), sa.ForeignKey("teachers.teacher_id"), nullable=False),
        sa.Column("observed_at", sa.DateTime(), nullable=False),
        sa.Column("setting", sa.String(), nullable=False),
        sa.Column("objective_text", sa.Text(), nullable=False),
        sa.Column("educator_actions", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_request_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    for column in ("centre_id", "class_id", "author_teacher_id", "observed_at", "status", "source_request_id", "created_at"):
        op.create_index(f"ix_observations_{column}", "observations", [column])

    op.create_table(
        "observation_children",
        sa.Column("link_id", sa.String(), primary_key=True),
        sa.Column("observation_id", sa.String(), sa.ForeignKey("observations.observation_id", ondelete="CASCADE"), nullable=False),
        sa.Column("child_id", sa.String(), sa.ForeignKey("children.child_id"), nullable=False),
        sa.UniqueConstraint("observation_id", "child_id", name="uq_observation_child"),
    )
    op.create_index("ix_observation_children_observation_id", "observation_children", ["observation_id"])
    op.create_index("ix_observation_children_child_id", "observation_children", ["child_id"])

    op.create_table(
        "educational_records",
        sa.Column("record_id", sa.String(), primary_key=True),
        sa.Column("centre_id", sa.String(), sa.ForeignKey("centres.centre_id"), nullable=False),
        sa.Column("class_id", sa.String(), sa.ForeignKey("classes.class_id"), nullable=False),
        sa.Column("author_teacher_id", sa.String(), sa.ForeignKey("teachers.teacher_id"), nullable=False),
        sa.Column("record_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("analysis", sa.Text(), nullable=False),
        sa.Column("curriculum_links", sa.JSON(), nullable=False),
        sa.Column("next_steps", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source_request_id", sa.String(), nullable=True),
        sa.Column("idempotency_key", sa.String(), nullable=False, unique=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("approved_by", sa.String(), nullable=True),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    for column in ("centre_id", "class_id", "author_teacher_id", "record_type", "status", "source_request_id", "created_at"):
        op.create_index(f"ix_educational_records_{column}", "educational_records", [column])

    op.create_table(
        "educational_record_observations",
        sa.Column("link_id", sa.String(), primary_key=True),
        sa.Column("record_id", sa.String(), sa.ForeignKey("educational_records.record_id", ondelete="CASCADE"), nullable=False),
        sa.Column("observation_id", sa.String(), sa.ForeignKey("observations.observation_id"), nullable=False),
        sa.UniqueConstraint("record_id", "observation_id", name="uq_educational_record_observation"),
    )
    op.create_index("ix_educational_record_observations_record_id", "educational_record_observations", ["record_id"])
    op.create_index("ix_educational_record_observations_observation_id", "educational_record_observations", ["observation_id"])

    op.create_table(
        "record_exports",
        sa.Column("export_id", sa.String(), primary_key=True),
        sa.Column("teacher_id", sa.String(), sa.ForeignKey("teachers.teacher_id"), nullable=False),
        sa.Column("record_ids", sa.JSON(), nullable=False),
        sa.Column("format", sa.String(), nullable=False),
        sa.Column("template_name", sa.String(), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
    )
    for column in ("teacher_id", "status", "created_at"):
        op.create_index(f"ix_record_exports_{column}", "record_exports", [column])

    op.create_table(
        "tool_action_requests",
        sa.Column("action_id", sa.String(), primary_key=True),
        sa.Column("request_id", sa.String(), nullable=False, unique=True),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("teacher_id", sa.String(), nullable=False),
        sa.Column("class_id", sa.String(), nullable=True),
        sa.Column("tool_name", sa.String(), nullable=False),
        sa.Column("arguments", sa.JSON(), nullable=False),
        sa.Column("arguments_hash", sa.String(), nullable=False),
        sa.Column("preview", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("executed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    for column in ("request_id", "session_id", "teacher_id", "class_id", "tool_name", "status", "expires_at"):
        op.create_index(f"ix_tool_action_requests_{column}", "tool_action_requests", [column])

    op.create_table(
        "knowledge_sources",
        sa.Column("source_id", sa.String(), primary_key=True),
        sa.Column("source_key", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("effective_date", sa.DateTime(), nullable=True),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.Column("knowledge_scope", sa.String(), nullable=False),
        sa.Column("index_version", sa.String(), nullable=False),
        sa.Column("index_status", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    for column in ("source_key", "source_type", "knowledge_scope", "index_status"):
        op.create_index(f"ix_knowledge_sources_{column}", "knowledge_sources", [column], unique=column == "source_key")

    op.create_table(
        "audit_events",
        sa.Column("audit_id", sa.String(), primary_key=True),
        sa.Column("teacher_id", sa.String(), nullable=True),
        sa.Column("class_id", sa.String(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("resource_type", sa.String(), nullable=False),
        sa.Column("resource_id", sa.String(), nullable=True),
        sa.Column("tool_name", sa.String(), nullable=True),
        sa.Column("result", sa.String(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    for column in ("teacher_id", "class_id", "action", "resource_type", "resource_id", "created_at"):
        op.create_index(f"ix_audit_events_{column}", "audit_events", [column])

    op.add_column("long_term_memories", sa.Column("centre_id", sa.String(), nullable=True))
    op.add_column("long_term_memories", sa.Column("confidence", sa.Float(), nullable=False, server_default="1"))
    op.add_column("long_term_memories", sa.Column("review_status", sa.String(), nullable=False, server_default="auto"))
    op.add_column("long_term_memories", sa.Column("source_request_id", sa.String(), nullable=True))
    op.add_column("long_term_memories", sa.Column("last_confirmed_at", sa.DateTime(), nullable=True))
    for column in ("centre_id", "review_status", "source_request_id"):
        op.create_index(f"ix_long_term_memories_{column}", "long_term_memories", [column])


def downgrade() -> None:
    for column in ("source_request_id", "review_status", "centre_id"):
        op.drop_index(f"ix_long_term_memories_{column}", table_name="long_term_memories")
    for column in ("last_confirmed_at", "source_request_id", "review_status", "confidence", "centre_id"):
        op.drop_column("long_term_memories", column)
    for table in (
        "audit_events",
        "knowledge_sources",
        "tool_action_requests",
        "record_exports",
        "educational_record_observations",
        "educational_records",
        "observation_children",
        "observations",
        "children",
        "teacher_class_memberships",
        "classes",
        "teachers",
        "centres",
    ):
        op.drop_table(table)
