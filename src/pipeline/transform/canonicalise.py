"""Canonicalisation: one source record in, one clean record out — then merged.

Two steps, deliberately separate.

``canonicalise`` turns a ``RawBook`` into a ``CleanBook``: normalised
comparison forms, validated ISBN, parsed year, canonical identity. It is per
record and knows nothing about other sources.

``merge_candidates`` decides what the catalogue says when several sources
describe the same book and disagree. It is order-independent by construction,
because the load stage recomputes canonical fields on every ingest and a result
that depended on input order would rewrite rows that had not changed.
"""

from __future__ import annotations

from typing import Any

from pipeline.extract.base import Rejected
from pipeline.models.domain import CleanBook, RawBook, SourceName
from pipeline.transform.identity import identity_key
from pipeline.transform.isbn import select_canonical_isbn
from pipeline.transform.normalise import (
    normalise_author,
    normalise_subject,
    normalise_title,
    parse_year,
    select_language,
)

# Applied only when two records look equally complete. Open Library has the
# best bibliographic metadata, Google Books is a useful third opinion, and
# Gutendex is authoritative about Project Gutenberg and little else.
SOURCE_PRIORITY = {
    SourceName.OPENLIBRARY: 0,
    SourceName.GOOGLEBOOKS: 1,
    SourceName.GUTENDEX: 2,
}

# Fields resolved independently across candidates. Excludes identity, which is
# shared by definition, and provenance, which belongs to the winning record.
_MERGEABLE_FIELDS = (
    "title",
    "normalised_title",
    "subtitle",
    "isbn13",
    "published_year",
    "page_count",
    "language",
    "publisher",
    "description",
    "cover_url",
    "download_count",
    "normalised_first_author",
)

# Contribute to the completeness score. Identity and provenance are excluded:
# every record has them, so counting them would flatten the comparison.
_COMPLETENESS_FIELDS = (
    *_MERGEABLE_FIELDS,
    "authors",
    "subjects",
)


def canonicalise(record: RawBook) -> CleanBook | Rejected:
    """Normalise and validate one source record.

    Returns a ``Rejected`` rather than raising when the record cannot be given
    an identity — a book with no usable title cannot be addressed, updated or
    deduplicated, so admitting it would put an unreachable row in the
    catalogue.
    """
    normalised_title = normalise_title(record.title)
    if normalised_title is None:
        return Rejected(
            source=record.source,
            source_id=record.source_id,
            raw_payload=record.raw_payload,
            rejection_code="unidentifiable",
            detail="title is empty once normalised, so no identity can be derived",
        )

    first_author = normalise_author(record.authors[0].name) if record.authors else None
    isbn13 = select_canonical_isbn(record.isbns)
    published_year = parse_year(record.published)

    return CleanBook(
        source=record.source,
        source_id=record.source_id,
        identity_key=identity_key(
            isbn13=isbn13,
            title=normalised_title,
            first_author=first_author,
            year=published_year,
        ),
        title=record.title,
        normalised_title=normalised_title,
        subtitle=record.subtitle,
        isbn13=isbn13,
        published_year=published_year,
        page_count=record.page_count,
        language=select_language(record.languages),
        publisher=record.publisher,
        description=record.description,
        cover_url=record.cover_url,
        download_count=record.download_count,
        authors=list(record.authors),
        normalised_first_author=first_author,
        subjects=[subject for subject in record.subjects if normalise_subject(subject) is not None],
        source_updated=record.source_updated,
        raw_payload=record.raw_payload,
    )


def _completeness(book: CleanBook) -> int:
    """How many usable fields a record carries."""
    return sum(1 for field in _COMPLETENESS_FIELDS if getattr(book, field))


def _rank(book: CleanBook) -> tuple[int, int, str]:
    """Sort key: most complete first, then source priority, then source id.

    The trailing source id makes the order total. Without it two records that
    tie on both earlier keys would sort by input order, and the merge would
    stop being reproducible.
    """
    return (-_completeness(book), SOURCE_PRIORITY[book.source], book.source_id)


def merge_candidates(candidates: list[CleanBook]) -> CleanBook:
    """Resolve several records for one book into the canonical version.

    Field by field, the first candidate in rank order that has a non-null value
    wins. That ordering matters: priority decides between two *answers*, never
    between an answer and none, so a sparse high-priority source cannot blank a
    field a lower-priority one supplied.

    Provider timestamps are deliberately not consulted. ``source_updated``
    means something different at each provider and Gutendex publishes none at
    all, so "newer" is not comparable across sources and would make the result
    depend on which provider happened to touch a record last.

    Raises:
        ValueError: the list is empty, or the candidates do not share one
            canonical identity — merging across identities would fuse two
            different books into a row nothing could separate again.
    """
    if not candidates:
        msg = "merge_candidates requires at least one candidate"
        raise ValueError(msg)

    identities = {book.identity_key for book in candidates}
    if len(identities) > 1:
        msg = f"candidates span {len(identities)} identity keys: {sorted(identities)}"
        raise ValueError(msg)

    ordered = sorted(candidates, key=_rank)
    winner = ordered[0]

    resolved: dict[str, Any] = {}
    for field in _MERGEABLE_FIELDS:
        for book in ordered:
            value = getattr(book, field)
            if value is not None:
                resolved[field] = value
                break

    # Authors and subjects come from the winning record as a set rather than
    # field by field: an author list spliced from two providers would mix
    # orderings and produce a credit order neither source ever published.
    richest_authors = next((b for b in ordered if b.authors), winner)
    richest_subjects = next((b for b in ordered if b.subjects), winner)

    return winner.model_copy(
        update={
            **resolved,
            "authors": list(richest_authors.authors),
            "subjects": list(richest_subjects.subjects),
        }
    )
