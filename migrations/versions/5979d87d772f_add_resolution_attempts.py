"""Add resolution_attempts.

One row per source attempt per candidate. With an unofficial primary source,
knowing which source answered and why the others were not used is operational
data rather than debug noise: without it, a run that fell back for every
candidate is indistinguishable from one that never needed to.

Revision ID: 5979d87d772f
Revises: 870f7a5e1908
Create Date: 2026-08-11 12:31:59.392457

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5979d87d772f"
down_revision: Union[str, Sequence[str], None] = "870f7a5e1908"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "resolution_attempts",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("candidate_key", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("attempt_no", sa.SmallInteger(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("fallback_reason", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "outcome IN ('resolved', 'partial', 'no_match', 'contract_failure', 'unavailable', 'skipped')",
            name=op.f("ck_resolution_attempts_outcome_known"),
        ),
        sa.CheckConstraint(
            "source IN ('goodreads', 'openlibrary', 'googlebooks', 'gutendex')",
            name=op.f("ck_resolution_attempts_source_known"),
        ),
        sa.CheckConstraint(
            "attempt_no > 0", name=op.f("ck_resolution_attempts_attempt_no_positive")
        ),
        sa.CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name=op.f("ck_resolution_attempts_duration_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["ingestion_runs.id"],
            name=op.f("fk_resolution_attempts_run_id_ingestion_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id", "candidate_key", "source", "attempt_no", name=op.f("pk_resolution_attempts")
        ),
    )
    op.create_index(
        "idx_resolution_attempts_run_outcome",
        "resolution_attempts",
        ["run_id", "source", "outcome"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_resolution_attempts_run_outcome", table_name="resolution_attempts")
    op.drop_table("resolution_attempts")
