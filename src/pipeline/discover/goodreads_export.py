"""Candidates from a Goodreads export produced elsewhere.

A file of scraped Goodreads records is discovery, not a live source: it says
"these books exist and here is enough to look them up", which is the same claim
the Open Library dump makes. So it enters the pipeline the same way — as
candidates carrying a retained payload — and everything downstream is
unchanged.

Two things make this worth having beyond volume. The export was gathered once,
so resolving 32,000 books from it costs Goodreads nothing at all, which is a
better answer to a source whose terms restrict automated collection than any
rate limit. And its ``author`` is a genuine list, where the search card this
pipeline scrapes carries one — so a book with three contributors arrives with
three.

The payload is translated into the shape the Goodreads mapper already reads
rather than teaching that mapper a second dialect. One reader, one contract:
whatever the loader replays later must be the same shape whether the record
came from this file or from a live search.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from pipeline.models.domain import CandidateBook, SourceName

logger = structlog.get_logger(__name__)


def _authors(record: dict[str, Any]) -> list[str]:
    """Every credited author, in order, skipping blanks."""
    raw = record.get("author")
    entries = raw if isinstance(raw, list) else [raw]
    names: list[str] = []
    for entry in entries:
        if isinstance(entry, str) and entry.strip() and entry.strip() not in names:
            names.append(entry.strip())
    return names


def to_goodreads_payload(record: dict[str, Any]) -> dict[str, Any] | None:
    """Translate one export record into the shape the Goodreads mapper reads.

    Returns ``None`` when the record cannot identify a book. A row without an
    id or a title is not a thin record, it is not a record.
    """
    book_id = record.get("goodreadsId")
    title = record.get("title")
    if not book_id or not isinstance(title, str) or not title.strip():
        return None

    names = _authors(record)
    payload: dict[str, Any] = {
        "bookId": str(book_id),
        "title": title.strip(),
        "bookTitleBare": title.strip(),
        # The mapper reads a single author object, as the search card supplies.
        # The full list is kept alongside it so nothing is discarded at the
        # door; the mapper can be taught to read it without another export.
        "author": {"name": names[0]} if names else {},
        "authors": [{"name": name} for name in names],
        "imageUrl": record.get("coverUrl"),
        "avgRating": str(record["rating"]) if record.get("rating") is not None else None,
        "ratingsCount": record.get("ratingCount"),
        "bookUrl": record.get("goodreadsUrl"),
        # Marks where this came from. A record replayed years from now should
        # say whether a person watched it arrive or a file supplied it.
        "_export": {"source_file": record.get("sourceFile"), "position": record.get("position")},
    }
    return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ExportReport:
    """What an export file is, before anything tries to resolve it.

    A file that is the wrong shape should say so in seconds, not become an
    empty run that took an hour and looked like it worked.
    """

    path: Path
    total: int = 0
    usable: int = 0
    problem: str | None = None
    missing_id: int = 0
    missing_title: int = 0
    sample_keys: tuple[str, ...] = ()

    @property
    def compatible(self) -> bool:
        return self.problem is None and self.usable > 0

    def explain(self) -> str:
        """One line an operator can act on."""
        if self.problem:
            return f"{self.path.name}: {self.problem}"
        if not self.usable:
            return (
                f"{self.path.name}: {self.total} records, none usable. "
                f"{self.missing_id} lack goodreadsId, {self.missing_title} lack title. "
                f"Keys seen: {', '.join(self.sample_keys) or 'none'}"
            )
        return f"{self.path.name}: {self.usable} of {self.total} records usable" + (
            f", {self.total - self.usable} skipped" if self.total > self.usable else ""
        )


def inspect_export(path: Path) -> ExportReport:
    """Read an export and report whether it can be ingested.

    Deliberately reads the whole file rather than sampling. A record that
    breaks the reader is more likely at the end than the beginning — an export
    is usually appended to — and the file is small enough that certainty costs
    a second.
    """
    if not path.exists():
        return ExportReport(path=path, problem="file not found")

    try:
        records = list(_records(path))
    except json.JSONDecodeError as error:
        return ExportReport(path=path, problem=f"not valid JSON: {error}")
    except ValueError as error:
        return ExportReport(path=path, problem=str(error))

    usable = missing_id = missing_title = 0
    keys: set[str] = set()
    for record in records:
        keys.update(record)
        if not record.get("goodreadsId"):
            missing_id += 1
        elif not str(record.get("title") or "").strip():
            missing_title += 1
        else:
            usable += 1

    return ExportReport(
        path=path,
        total=len(records),
        usable=usable,
        missing_id=missing_id,
        missing_title=missing_title,
        sample_keys=tuple(sorted(keys)[:12]),
    )


def _records(path: Path) -> Iterator[dict[str, Any]]:
    """Every object in the export.

    Read whole rather than streamed: the file is a single JSON array, so there
    is no line-oriented shortcut, and 14 MB is a fraction of what resolving the
    same books will hold anyway.
    """
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)

    if isinstance(loaded, dict):
        # Some exports wrap the list; take the first list-valued key rather
        # than guessing at a name.
        loaded = next((value for value in loaded.values() if isinstance(value, list)), [])
    if not isinstance(loaded, list):
        msg = f"expected a list of records, got {type(loaded).__name__}"
        raise ValueError(msg)

    for entry in loaded:
        if isinstance(entry, dict):
            yield entry


def stream_candidates(
    path: Path, *, max_candidates: int | None = None, start_index: int = 0
) -> Iterator[CandidateBook]:
    """Yield candidates from the export, resuming at ``start_index``.

    ``start_index`` counts records read, not candidates emitted, so a resumed
    run continues where the last one stopped rather than where it succeeded —
    the same distinction the dump reader draws, for the same reason.
    """
    emitted = 0
    skipped = 0
    for index, record in enumerate(_records(path)):
        if max_candidates is not None and emitted >= max_candidates:
            break
        if index < start_index:
            continue

        payload = to_goodreads_payload(record)
        if payload is None:
            skipped += 1
            continue

        yield CandidateBook(
            candidate_key=f"goodreads:{payload['bookId']}",
            title=payload["title"],
            authors=_authors(record),
            discovery_payload=payload,
            discovery_source=SourceName.GOODREADS,
        )
        emitted += 1

    logger.info(
        "discovery.goodreads_export_read",
        path=str(path),
        emitted=emitted,
        unusable=skipped,
        start_index=start_index,
    )


def build_manifest(
    path: Path, manifest_path: Path, *, max_candidates: int | None = None, start_index: int = 0
) -> int:
    """Write a candidate manifest from the export, and say how many.

    The manifest format is the dump's, so the resolver, the DAG and every test
    downstream cannot tell which discovery source produced it.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    written = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in stream_candidates(
            path, max_candidates=max_candidates, start_index=start_index
        ):
            handle.write(json.dumps(candidate.model_dump(mode="json")) + "\n")
            written += 1

    # Atomic: a crash mid-write must not leave a half-manifest that the next
    # task reads as complete.
    temporary.replace(manifest_path)
    logger.info("discovery.goodreads_manifest_written", candidates=written, path=str(manifest_path))
    return written
