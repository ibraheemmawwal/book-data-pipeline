"""Re-resolving contested books through a tie-breaker source.

Most books in the catalogue are corroborated. A minority are contested — two
documented sources reported different values for the same field — and those are
the only ones worth spending a restricted source on.

That framing is the whole justification for this module. Running Goodreads
across the catalogue would be bulk collection; running it against the handful
of books where documented sources genuinely conflict is a targeted lookup with
a stated reason, bounded by a count this module enforces rather than hopes for.

It changes nothing about the source's terms, which restrict automated
collection irrespective of volume. The double gate stays exactly as it is: this
refuses to run unless both are set deliberately.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import Engine, text

from pipeline.config import Settings
from pipeline.db import build_engine
from pipeline.extract.goodreads import (
    GoodreadsExtractor,
    GoodreadsNotAcceptedError,
    GoodreadsUnavailableError,
)
from pipeline.load import CatalogueLoader
from pipeline.models.domain import CandidateBook, CleanBook
from pipeline.observability.runs import finalise_run, start_run
from pipeline.transform import canonicalise

logger = structlog.get_logger(__name__)

# Fields whose disagreement is worth a tie-breaker. Deliberately the same list
# the API compares on: a book the catalogue calls contested and a book this
# module re-resolves must be the same set, or the tool and the pipeline are
# describing different things.
COMPARABLE_KEYS: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "published_year": ("published", "first_publish_year", "publish_date", "publishedDate"),
    "publisher": ("publisher", "publishers"),
    "page_count": ("page_count", "number_of_pages", "number_of_pages_median", "pageCount"),
}
NESTED_ROOTS = ("", "volumeInfo")


@dataclass
class ContestedReport:
    """What one tie-breaking run did."""

    examined: int = 0
    contested: int = 0
    queried: int = 0
    resolved: int = 0
    unresolved: int = 0
    loaded: int = 0
    run_id: UUID | None = None
    errors: list[str] = field(default_factory=list)


def _readable(payload: Any) -> dict[str, Any]:
    """Flatten the roots a source may nest its fields under."""
    if not isinstance(payload, dict):
        return {}
    merged: dict[str, Any] = {}
    for root in reversed(NESTED_ROOTS):
        section = payload if root == "" else payload.get(root)
        if isinstance(section, dict):
            merged.update(section)
    return merged


def conflict_count(payloads: list[Any]) -> int:
    """How many fields these sources report differently."""
    conflicts = 0
    for keys in COMPARABLE_KEYS.values():
        reported: set[str] = set()
        for payload in payloads:
            flat = _readable(payload)
            for key in keys:
                value = flat.get(key)
                if value in (None, "", []):
                    continue
                if isinstance(value, list):
                    value = value[0] if value else None
                if value is not None:
                    reported.add(str(value).strip().lower())
                break
        if len(reported) > 1:
            conflicts += 1
    return conflicts


def find_contested(engine: Engine, *, minimum_conflicts: int, limit: int) -> list[dict[str, Any]]:
    """Books whose sources conflict on at least ``minimum_conflicts`` fields.

    Ordered by conflict count so a bounded run spends its budget on the worst
    records rather than the first ones it happens to see.
    """
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                """
                SELECT b.id, b.title, b.isbn13, b.identity_key,
                       array_agg(bs.raw_payload) AS payloads,
                       array_agg(DISTINCT bs.source) AS sources
                FROM books AS b
                JOIN book_sources AS bs ON bs.book_id = b.id
                GROUP BY b.id, b.title, b.isbn13
                HAVING count(DISTINCT bs.source) > 1
                """
            )
        ).all()

    contested = []
    for row in rows:
        count = conflict_count(list(row.payloads))
        if count >= minimum_conflicts:
            contested.append(
                {
                    "id": row.id,
                    "title": row.title,
                    "isbn13": row.isbn13,
                    "identity_key": row.identity_key,
                    "conflicts": count,
                    "sources": list(row.sources),
                }
            )

    contested.sort(key=lambda item: item["conflicts"], reverse=True)
    return contested[:limit]


def _attach_to(cleaned: CleanBook, book: dict[str, Any]) -> CleanBook | None:
    """Re-key an observation onto an existing book's identity.

    identity_key and isbn13 must move together — a fallback identity with an
    ISBN present, or an ISBN identity naming a different one, merges the wrong
    books, which is why the model rejects it.

    Both values here come from the same row, which the loader already wrote
    consistently, so the pair is reconcilable by construction. The check is
    kept anyway: it costs nothing and the alternative to noticing a violation
    is silently merging two different books, which no later query can undo.

    A full re-validation is not available — content_hash is computed, so a
    dump-and-revalidate round trip rejects its own output.
    """
    identity, isbn13 = book["identity_key"], book["isbn13"]

    if identity.startswith("isbn:"):
        if identity.removeprefix("isbn:") != isbn13:
            return None
    elif identity.startswith("fallback:"):
        if isbn13 is not None:
            return None
    else:
        return None

    return cleaned.model_copy(update={"identity_key": identity, "isbn13": isbn13})


async def _resolve_one(extractor: GoodreadsExtractor, client: Any, book: dict[str, Any]) -> Any:
    candidate = CandidateBook(
        candidate_key=f"contested:{book['id']}",
        title=book["title"],
        isbns=[book["isbn13"]] if book["isbn13"] else [],
    )
    return await extractor.resolve(
        client, candidate.lookup_query(), isbn=candidate.preferred_isbn()
    )


async def _run(
    settings: Settings, engine: Engine, report: ContestedReport, books: list[Any]
) -> None:
    extractor = GoodreadsExtractor(settings)
    loader = CatalogueLoader()
    client = extractor.build_client()
    try:
        for book in books:
            if extractor.circuit_open:
                # Once it has refused us, stop for the run. Re-probing a source
                # that pushed back is exactly what the containment rules forbid.
                report.errors.append("circuit opened; stopping")
                break

            report.queried += 1
            try:
                observation = await _resolve_one(extractor, client, book)
            except GoodreadsUnavailableError as error:
                report.unresolved += 1
                report.errors.append(str(error)[:120])
                continue

            if observation is None:
                report.unresolved += 1
                continue

            cleaned = canonicalise(observation)
            if not isinstance(cleaned, CleanBook):
                report.unresolved += 1
                continue

            # Attach the observation to the book it was fetched for, rather
            # than letting identity resolution decide.
            #
            # This is the whole difference between enriching a record and
            # creating one. The tie-breaker was asked about a *known* book; if
            # its answer carries no ISBN it derives a fresh fallback identity,
            # the loader sees an unfamiliar record, and the contested book
            # quietly gains a duplicate instead of a third source. That is
            # exactly what happened on the first run: 20 books, 20 duplicates.
            attached = _attach_to(cleaned, book)
            if attached is None:
                report.unresolved += 1
                report.errors.append(f"could not attach observation for {book['title'][:40]}")
                continue

            report.resolved += 1
            assert report.run_id is not None  # opened before any query
            outcome = loader.load(engine, [attached], run_id=report.run_id)
            report.loaded += outcome.records_loaded
    finally:
        await client.aclose()


def resolve_contested(
    settings: Settings,
    *,
    minimum_conflicts: int = 2,
    limit: int = 50,
    engine: Engine | None = None,
) -> ContestedReport:
    """Re-resolve the most contested books through Goodreads.

    Raises:
        GoodreadsNotAcceptedError: either gate is unset. Both are deliberate
            acknowledgements, and a targeted run is not a reason to skip them.
    """
    active = engine or build_engine(settings.database_url)
    report = ContestedReport()

    # Before anything observable. A refused run should leave no run row behind
    # suggesting work was attempted, and the gate is cheap to check.
    GoodreadsExtractor(settings).ensure_accepted()

    books = find_contested(active, minimum_conflicts=minimum_conflicts, limit=limit)
    report.examined = len(books)
    report.contested = len(books)

    if not books:
        logger.info("contested.none_found", minimum_conflicts=minimum_conflicts)
        return report

    with active.begin() as connection:
        report.run_id = start_run(connection)

    try:
        asyncio.run(_run(settings, active, report, books))
    except GoodreadsNotAcceptedError:
        with active.begin() as connection:
            finalise_run(connection, report.run_id, status="failed")
        raise
    except Exception:
        with active.begin() as connection:
            finalise_run(connection, report.run_id, status="failed")
        raise

    with active.begin() as connection:
        finalise_run(
            connection,
            report.run_id,
            status="success" if report.resolved else "partial_success",
            records_extracted=report.resolved,
            records_loaded=report.loaded,
        )

    logger.info(
        "contested.complete",
        contested=report.contested,
        queried=report.queried,
        resolved=report.resolved,
    )
    return report
