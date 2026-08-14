"""Catalogue-side observability, read-only.

Runs inside a container because the catalogue URL and credentials live there.
Emits text by default and JSON on request, so the same numbers serve a terminal
and a dashboard without being gathered twice and drifting apart.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from sqlalchemy import text

from pipeline.config import Settings
from pipeline.db import build_engine

QUERIES: dict[str, str] = {
    "books": "SELECT count(*) FROM books",
    "book_sources": "SELECT count(*) FROM book_sources",
    "authors": "SELECT count(*) FROM authors",
    "series": "SELECT count(*) FROM series",
    "with_year": "SELECT count(*) FROM books WHERE published_year IS NOT NULL",
    "with_isbn": "SELECT count(*) FROM books WHERE isbn13 IS NOT NULL",
    "multi_source": (
        "SELECT count(*) FROM (SELECT book_id FROM book_sources "
        "GROUP BY book_id HAVING count(DISTINCT source) > 1) t"
    ),
    "multi_author": (
        "SELECT count(*) FROM (SELECT book_id FROM book_authors "
        "GROUP BY book_id HAVING count(*) > 1) t"
    ),
    "goodreads_enriched": (
        "SELECT count(*) FROM book_sources WHERE source='goodreads' AND raw_payload ? '_edition'"
    ),
    "goodreads_rows": "SELECT count(*) FROM book_sources WHERE source='goodreads'",
    "runs_open": "SELECT count(*) FROM ingestion_runs WHERE status IN ('running','processing')",
    "goodreads_pending": (
        "SELECT count(*) FROM book_sources WHERE source='goodreads' "
        "AND NOT (raw_payload ? '_edition') AND NOT (raw_payload ? '_detail')"
    ),
    # Minutes since Goodreads last refused us, or NULL. "Enrichment is doing
    # nothing" and "enrichment is deliberately waiting" look identical from the
    # book counts, and the second one is not a problem — but only if it says so.
    "goodreads_refused_min_ago": (
        "SELECT EXTRACT(EPOCH FROM (now() - max(finished_at)))/60 "
        "FROM source_runs WHERE source='goodreads' AND status='refused'"
    ),
}


def gather() -> dict[str, Any]:
    with build_engine(Settings().database_url).connect() as connection:
        scalars: dict[str, Any] = {}
        for name, sql in QUERIES.items():
            value = connection.execute(text(sql)).scalar()
            # "never refused" is genuinely absent, not zero — zero would read as
            # "refused just now", which is the opposite of the truth.
            scalars[name] = value if name.endswith("_min_ago") else (value or 0)
        sources = {
            row.source: row.n
            for row in connection.execute(
                text(
                    "SELECT source, count(*) AS n FROM book_sources GROUP BY source ORDER BY n DESC"
                )
            )
        }
        runs = [
            {"status": row.status, "extracted": row.records_extracted, "loaded": row.records_loaded}
            for row in connection.execute(
                text(
                    "SELECT status, records_extracted, records_loaded FROM ingestion_runs "
                    "ORDER BY started_at DESC LIMIT 5"
                )
            )
        ]
    return {**scalars, "sources": sources, "recent_runs": runs}


def main() -> None:
    data = gather()
    if "--json" in sys.argv:
        print(json.dumps(data))
        return

    books = max(1, data["books"])
    print(
        f"  books {data['books']}   sources {data['book_sources']}   "
        f"authors {data['authors']}   series {data['series']}"
    )
    print(
        f"  year coverage  {100 * data['with_year'] / books:.1f}%    "
        f"isbn coverage {100 * data['with_isbn'] / books:.1f}%"
    )
    print(f"  multi-source   {data['multi_source']}    multi-author {data['multi_author']}")
    gr = max(1, data["goodreads_rows"])
    print(
        f"  goodreads enriched {data['goodreads_enriched']}/{data['goodreads_rows']} "
        f"({100 * data['goodreads_enriched'] / gr:.0f}%)   "
        f"pending {data['goodreads_pending']}"
    )
    print(f"  runs still open    {data['runs_open']}")

    refused = data["goodreads_refused_min_ago"]
    if refused is None:
        print("  goodreads          no refusal on record")
    else:
        cooldown = Settings().goodreads_cooldown_minutes
        remaining = cooldown - refused
        state = f"cooling down, {remaining:.0f} min left" if remaining > 0 else "clear to resume"
        print(f"  goodreads          refused {refused:.0f} min ago — {state}")
    print("\n  rows per source:")
    for source, count in data["sources"].items():
        print(f"    {source:<14}{count}")


if __name__ == "__main__":
    main()
