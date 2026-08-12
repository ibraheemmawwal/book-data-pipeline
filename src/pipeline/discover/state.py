"""Where discovery got to.

Without this, every run reads the dump from the beginning and produces the same
candidates. A scheduled pipeline then re-resolves books it already holds and
never reaches the rest of the file — which is a repeated no-op wearing an
orchestrator's clothes.

The position is a line number rather than a byte offset. Bytes would be faster
to seek to, and wrong: the file is gzipped, so there is nothing to seek to
without decompressing anyway, and a byte offset landing mid-line would silently
corrupt the first record of every resumed run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from sqlalchemy import Connection, text

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveryPosition:
    """How far through one dump discovery has read."""

    dump_key: str
    line_offset: int
    candidates_emitted: int
    exhausted: bool


def dump_key(path: Path) -> str:
    """Identify a dump by name and size.

    Two dumps have entirely different content at the same line, so a position
    without one means nothing. Name alone is not enough — the published file is
    always ``ol_dump_editions_latest`` and its contents change monthly; adding
    the size makes a refreshed dump a different key, which resets the position
    rather than resuming into unrelated data.
    """
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return f"{path.name}:{size}"


def read_position(connection: Connection, key: str) -> DiscoveryPosition:
    """The stored position, or the start of the file."""
    row = connection.execute(
        text(
            """
            SELECT dump_key, line_offset, candidates_emitted, exhausted
            FROM discovery_state WHERE dump_key = :key
            """
        ),
        {"key": key},
    ).first()

    if row is None:
        return DiscoveryPosition(key, 0, 0, exhausted=False)
    return DiscoveryPosition(row.dump_key, row.line_offset, row.candidates_emitted, row.exhausted)


def save_position(
    connection: Connection,
    key: str,
    *,
    line_offset: int,
    candidates_emitted: int,
    exhausted: bool,
) -> None:
    """Record where this run stopped.

    Written after the candidates are durable, never before: a position saved
    first and then a crash would skip that slice of the dump forever, and
    nothing downstream would report a gap.
    """
    connection.execute(
        text(
            """
            INSERT INTO discovery_state
                (dump_key, line_offset, candidates_emitted, exhausted, updated_at)
            VALUES (:key, :line_offset, :emitted, :exhausted, now())
            ON CONFLICT (dump_key) DO UPDATE SET
                line_offset = EXCLUDED.line_offset,
                candidates_emitted = discovery_state.candidates_emitted
                                     + EXCLUDED.candidates_emitted,
                exhausted = EXCLUDED.exhausted,
                updated_at = now()
            """
        ),
        {
            "key": key,
            "line_offset": line_offset,
            "emitted": candidates_emitted,
            "exhausted": exhausted,
        },
    )
    logger.info(
        "discovery.position_saved",
        dump=key,
        line_offset=line_offset,
        exhausted=exhausted,
    )


def reset_position(connection: Connection, key: str) -> None:
    """Start this dump again from the beginning.

    For an operator who wants a full re-read without waiting for a new dump —
    re-ingesting is harmless because the load layer is idempotent.
    """
    connection.execute(text("DELETE FROM discovery_state WHERE dump_key = :key"), {"key": key})
