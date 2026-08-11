"""Series identity and normalisation.

A series name is a join key, so the bar is the same as for books: two spellings
of one series must collapse, and two different series must never merge. A wrong
merge attaches unrelated books to each other permanently and nothing downstream
can detect it.

Identity has two forms, and which one applies is a statement about evidence:

- ``srcseries:<source>:<id>`` when a source supplied a series id *and* that id
  was confirmed — for Goodreads, by the ``/series/`` slug matching the parsed
  name. This is the strong form.
- ``series:<sha256 of normalised name>`` otherwise. Source-independent by
  design, which is what lets two providers agree on one row without either
  having an id the other recognises.

An unconfirmed id is deliberately *not* used. It is a guess, and a guess must
not become the key everything else merges on.
"""

from __future__ import annotations

import hashlib
import html
from decimal import Decimal, InvalidOperation

from pipeline.models.domain import CleanSeriesMembership, RawSeriesMembership, SourceName
from pipeline.transform.normalise import _fold

SOURCE_SERIES_PREFIX = "srcseries:"
NAME_SERIES_PREFIX = "series:"

# book_series.position is NUMERIC(8,2): six digits before the point. A value
# the database would refuse must never reach it.
MAX_POSITION = Decimal("999999.99")


def normalise_series(value: str | None) -> str | None:
    """A series name's comparison form.

    Entities are decoded *before* folding, so an encoded ``&amp;`` cannot end
    up inside a join key and split one series into two.
    """
    if value is None:
        return None
    decoded = html.unescape(value)
    folded = _fold(decoded)
    return folded or None


def parse_series_position(value: str | None) -> Decimal | None:
    """Parse a series position as an exact decimal.

    Novellas really are numbered 0.5 and 2.5. Binary floating point would not
    compare equal to the value printed on the book, so positions stay
    ``Decimal`` from here all the way to ``NUMERIC(8,2)``.

    Returns ``None`` for anything unusable rather than raising: a bad position
    should cost the position, not the series relationship.
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        position = Decimal(text)
    except InvalidOperation:
        return None
    if position.is_nan() or position.is_infinite():
        return None
    if position < 0 or position > MAX_POSITION:
        return None
    return position


def series_identity_key(
    *,
    source: SourceName,
    source_series_id: str | None,
    normalised_name: str,
    confirmed: bool,
) -> str:
    """The canonical identity for one series.

    Raises:
        ValueError: there is neither a confirmed source id nor a usable name.
            Every unidentifiable series would otherwise collapse onto one row,
            merging the whole catalogue's series graph.
    """
    if confirmed and source_series_id:
        return f"{SOURCE_SERIES_PREFIX}{source.value}:{source_series_id}"

    if not normalised_name or not normalised_name.strip():
        msg = "a normalised series name is required when no confirmed source id exists"
        raise ValueError(msg)

    digest = hashlib.sha256(normalised_name.encode("utf-8")).hexdigest()
    return f"{NAME_SERIES_PREFIX}{digest}"


def canonicalise_series(
    membership: RawSeriesMembership, *, source: SourceName
) -> CleanSeriesMembership | None:
    """Validate and normalise one series relationship.

    Returns ``None`` when the name has no comparison form — an unnameable
    series cannot be joined on, and admitting it would put an unreachable row
    in the catalogue.
    """
    display = html.unescape(membership.name).strip()
    normalised = normalise_series(membership.name)
    if normalised is None or not display:
        return None

    return CleanSeriesMembership(
        identity_key=series_identity_key(
            source=source,
            source_series_id=membership.source_series_id,
            normalised_name=normalised,
            confirmed=membership.confirmed,
        ),
        name=display,
        normalised_name=normalised,
        source_series_id=membership.source_series_id,
        position=parse_series_position(membership.position),
        confirmed=membership.confirmed,
    )


def series_search_text(names: list[str]) -> str:
    """The deterministic projection stored on ``books.series_search_text``.

    Sorted and deduplicated because the loader recomputes it on every ingest;
    an order that wobbled would rewrite rows that had not changed and defeat
    the content-hash check. Returns ``""`` rather than ``None`` to match the
    NOT NULL DEFAULT '' column.
    """
    # Deduplicate on the *normalised* name, not the display form: "Discworld"
    # and "discworld" are one series and must contribute one term.
    by_normalised: dict[str, str] = {}
    for name in names:
        normalised = normalise_series(name)
        if normalised is None:
            continue
        display = html.unescape(name).strip()
        existing = by_normalised.get(normalised)
        if existing is None or display < existing:
            by_normalised[normalised] = display
    return " ".join(by_normalised[key] for key in sorted(by_normalised))
