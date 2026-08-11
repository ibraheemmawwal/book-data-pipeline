"""File-backed source and sink: the v1.0 implementation of the contracts.

JSONL because it is append-only, streamable and readable when something has
gone wrong at three in the morning. The same envelopes that later travel
through Kafka travel through here, so phase 2 changes the transport and
nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import structlog

from pipeline.models.events import (
    BookEvent,
    PartitionMarker,
    UnsupportedSchemaVersionError,
    decode_event,
)

logger = structlog.get_logger(__name__)

Event = BookEvent | PartitionMarker


class FileSink:
    """Appends events to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Touch on construction so a stage that emitted nothing still leaves
        # evidence it ran, rather than an absent file that reads as a crash.
        self._path.touch(exist_ok=True)
        self._written = 0

    def emit(self, records: Iterable[Event]) -> None:
        """Append events. Opens in append mode: a batch is not the whole run."""
        with self._path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(record.to_json().decode() + "\n")
                self._written += 1

    def flush(self) -> None:
        """A no-op for files, which are written on emit.

        Present because the contract promises it, and a caller should not have
        to know that this particular sink had nothing buffered.
        """
        logger.debug("file_sink.flush", path=str(self._path), written=self._written)


class FileSource:
    """Reads events back out of a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def commit(self) -> None:
        """No-op: a file has no consumer position to remember.

        Present so a stage can call it unconditionally and stay ignorant of
        which transport it is reading from.
        """

    def consume(self) -> Iterator[Event]:
        """Yield every event in the file, then stop.

        A missing file yields nothing rather than raising: a stage that has not
        run yet is empty, not broken.
        """
        if not self._path.exists():
            logger.info("file_source.absent", path=str(self._path))
            return

        with self._path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield decode_event(line.encode())
                except (
                    json.JSONDecodeError,
                    ValueError,
                    UnsupportedSchemaVersionError,
                ):
                    # One bad line costs that line. The alternative is a
                    # staging file that cannot be replayed at all.
                    #
                    # UnsupportedSchemaVersionError is listed explicitly: it is
                    # deliberately not a ValueError, so that a consumer can
                    # tell a future contract from a malformed payload and route
                    # them differently. Here there is no DLQ to route to, so
                    # both are skipped with a warning.
                    logger.warning(
                        "file_source.undecodable_line",
                        path=str(self._path),
                        line=number,
                    )
