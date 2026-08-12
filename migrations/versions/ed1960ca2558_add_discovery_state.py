"""Add discovery state.

Discovery had no memory. Every run read the dump from the beginning and
produced the same candidates, so a scheduled pipeline re-resolved books it
already held and never reached the rest of the file — which makes an
orchestrator a repeated no-op rather than something that makes progress.

One row per dump, holding the line the last run stopped at.

Revision ID: ed1960ca2558
Revises: 5979d87d772f
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "ed1960ca2558"
down_revision = "5979d87d772f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "discovery_state",
        # The dump identifies the position: two dumps have different content at
        # the same line, so an offset without one means nothing.
        sa.Column("dump_key", sa.Text(), primary_key=True),
        sa.Column("line_offset", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("candidates_emitted", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("exhausted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("discovery_state")
