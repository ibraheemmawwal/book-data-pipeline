"""Canonical identity and content hashing.

Identity decides which source records become the same book, and it must be a
pure function of content: the same record on a different day, in a different
process, with a differently-ordered JSON payload has to produce the same key.
If it does not, re-running the pipeline creates duplicate rows instead of
updating existing ones, and the idempotency the whole design rests on is gone.

Two identities, as set out in the TRD:

- **Ingestion identity** ``(source, source_id)`` — always present, and the
  conflict target for source rows. Not computed here; it comes from the source.
- **Canonical identity** — a validated ISBN-13 where one exists, otherwise a
  deterministic digest of normalised title, first author and year.

The fallback is an explicit heuristic. Two different books that share a
normalised title, author and year will merge, and one book catalogued under a
variant title will not. Both are documented limitations rather than bugs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

ISBN_PREFIX = "isbn:"
FALLBACK_PREFIX = "fallback:"

# A separator that cannot occur in a normalised field. Without one, "ab" + "c"
# and "a" + "bc" hash identically and merge two unrelated books; normalisation
# collapses whitespace but never strips a NUL, so no field can forge a boundary.
_FIELD_SEPARATOR = "\x00"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fallback_identity_key(
    normalised_title: str, normalised_first_author: str | None, year: int | None
) -> str:
    """Identity for a record with no usable ISBN.

    Absent author and year are encoded as empty strings rather than skipped, so
    a record missing an author is distinguishable from one whose author is the
    empty string — collapsing those would merge unrelated books.

    Raises:
        ValueError: the title is blank. Every ISBN-less record would otherwise
            collapse onto a single key, merging the entire catalogue.
    """
    if not normalised_title or not normalised_title.strip():
        msg = "a normalised title is required to build a fallback identity key"
        raise ValueError(msg)

    parts = (
        normalised_title,
        normalised_first_author or "",
        str(year) if year is not None else "",
    )
    return FALLBACK_PREFIX + _digest(_FIELD_SEPARATOR.join(parts))


def identity_key(
    *,
    isbn13: str | None,
    title: str,
    first_author: str | None,
    year: int | None,
) -> str:
    """The canonical identity for one record.

    An ISBN wins outright and the other fields are ignored: two sources
    describing the same ISBN must agree even when their titles are punctuated
    differently or their years disagree, which is exactly the case the fallback
    cannot handle.
    """
    if isbn13:
        return f"{ISBN_PREFIX}{isbn13}"
    return fallback_identity_key(title, first_author, year)


def _canonical_json(value: Any) -> str:
    """Serialise deterministically.

    ``sort_keys`` because providers make no promise about key order and a hash
    that moved with it would report every unchanged record as changed on every
    run. Lists keep their order — author one is not author two.
    """
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def payload_hash(payload: dict[str, Any]) -> str:
    """Fingerprint a raw source payload.

    Stored on ``book_sources`` so a re-ingested record that has not changed can
    be recognised without diffing every column.
    """
    return _digest(_canonical_json(payload))


def content_hash(fields: dict[str, Any]) -> str:
    """Fingerprint the canonical fields chosen for a book.

    Nulls are dropped before hashing, so a field a source stopped sending and
    one it sends as null are the same canonical state. Keeping them distinct
    would make ``books.updated_at`` move on a record that did not change, which
    is the very thing the idempotency test asserts against.
    """
    present = {k: v for k, v in fields.items() if v is not None}
    return _digest(_canonical_json(present))
