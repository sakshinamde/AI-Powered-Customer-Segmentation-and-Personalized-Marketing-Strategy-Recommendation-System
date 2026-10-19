"""Create SmartSeg tables.

Revision ID: 0001_initial
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("users", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("name", sa.String(length=255), nullable=False), sa.Column("email", sa.String(length=255), nullable=False), sa.Column("password_hash", sa.String(length=255), nullable=False), sa.Column("role", sa.String(length=50)), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_users_id", "users", ["id"])
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_table("datasets", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False), sa.Column("filename", sa.String(length=255), nullable=False), sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False), sa.Column("row_count", sa.Integer()), sa.Column("columns", sa.JSON(), nullable=False))
    op.create_index("ix_datasets_id", "datasets", ["id"])
    op.create_table("customers", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False), sa.Column("customer_id", sa.String(length=64), nullable=False), sa.Column("feature_values", sa.JSON(), nullable=False), sa.Column("raw_row", sa.JSON(), nullable=False))
    op.create_index("ix_customers_id", "customers", ["id"])
    op.create_table("cluster_runs", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False), sa.Column("algorithm", sa.String(length=30), nullable=False), sa.Column("k", sa.Integer()), sa.Column("params", sa.JSON(), nullable=False), sa.Column("silhouette_score", sa.Float(), nullable=False), sa.Column("n_clusters", sa.Integer()), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_cluster_runs_id", "cluster_runs", ["id"])
    op.create_table("cluster_assignments", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("cluster_run_id", sa.Integer(), sa.ForeignKey("cluster_runs.id"), nullable=False), sa.Column("customer_id", sa.Integer(), sa.ForeignKey("customers.id"), nullable=False), sa.Column("cluster_label", sa.Integer(), nullable=False), sa.Column("proj_x", sa.Float()), sa.Column("proj_y", sa.Float()))
    op.create_index("ix_cluster_assignments_id", "cluster_assignments", ["id"])
    op.create_table("strategies", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("dataset_id", sa.Integer(), sa.ForeignKey("datasets.id"), nullable=False), sa.Column("cluster_label", sa.Integer(), nullable=False), sa.Column("title", sa.String(length=255), nullable=False), sa.Column("recommendation", sa.Text(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_strategies_id", "strategies", ["id"])


def downgrade() -> None:
    op.drop_table("strategies")
    op.drop_table("cluster_assignments")
    op.drop_table("cluster_runs")
    op.drop_table("customers")
    op.drop_table("datasets")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_id", table_name="users")
    op.drop_table("users")
