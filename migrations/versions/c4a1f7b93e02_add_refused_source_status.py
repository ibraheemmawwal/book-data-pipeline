"""Allow a source_runs row to record a refusal.

The circuit breaker lives in one process and every scheduled task is a fresh
process, so a run that Goodreads blocked taught the next run nothing: it
started clean an hour later, walked into the same block and stopped in the same
place. Three consecutive runs did exactly that.

Fixing it needs somewhere durable to write "the source refused us, at this
time", and source_runs already carries per-source outcomes. The only thing in
the way was the status check, which allowed our decisions but not theirs:
"skipped" means we chose not to ask, and a refusal is not that.

Revision ID: c4a1f7b93e02
Revises: ed1960ca2558
"""

from __future__ import annotations

from alembic import op

revision = "c4a1f7b93e02"
down_revision = "ed1960ca2558"
branch_labels = None
depends_on = None

OLD = ("running", "success", "skipped", "failed")
NEW = ("running", "success", "skipped", "refused", "failed")


def _rewrite(values: tuple[str, ...]) -> None:
    """Swap the allowed-status list.

    ``op.f`` on both sides: the naming convention would otherwise expand an
    already-final name into ``ck_source_runs_ck_source_runs_status_known`` and
    the drop would miss.
    """
    rendered = ", ".join(f"'{value}'" for value in values)
    op.drop_constraint(op.f("ck_source_runs_status_known"), "source_runs", type_="check")
    op.create_check_constraint(
        op.f("ck_source_runs_status_known"), "source_runs", f"status IN ({rendered})"
    )


def upgrade() -> None:
    _rewrite(NEW)


def downgrade() -> None:
    # Rows written under the wider constraint would violate the narrower one.
    # Reclassifying them as "failed" is honest: under the old vocabulary a
    # refusal is a source that did not complete, and it keeps the downgrade
    # from failing on real data.
    op.execute("UPDATE source_runs SET status = 'failed' WHERE status = 'refused'")
    _rewrite(OLD)
