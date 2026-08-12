"""Discovering candidates from Gutendex.

An alternative to the Open Library dump, not a replacement for it. Gutendex
indexes Project Gutenberg, which is public-domain only — roughly 75,000 works,
almost all pre-1929. That makes it excellent for the classics and blind to
everything published since, which is why the dump stays the default: it is the
only source here with broad coverage of the last century.

Worth having anyway. It needs no 12 GB download, no credentials and no resume
position, so it is the fastest way to get a real catalogue standing from
nothing — and every book it yields is one three sources can later disagree
about.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import structlog

from pipeline.config import Settings
from pipeline.models.domain import CandidateBook

logger = structlog.get_logger(__name__)

PAGE_SIZE = 32  # Gutendex's own page size; asking for more changes nothing.


def _to_candidate(book: dict[str, Any]) -> CandidateBook | None:
    """One Gutendex book as a candidate, or None if it cannot be looked up.

    Same bar as the dump: a title plus something to search *with*. Gutendex
    always carries author names, which is why it clears the bar more often than
    a dump edition does.
    """
    title = book.get("title")
    identifier = book.get("id")
    if not isinstance(title, str) or not title.strip() or identifier is None:
        return None

    authors = [
        a["name"]
        for a in book.get("authors", [])
        if isinstance(a, dict) and isinstance(a.get("name"), str)
    ]
    if not authors:
        return None

    return CandidateBook(
        candidate_key=f"gutendex:{identifier}",
        title=title.strip(),
        authors=authors[:5],
        languages=[code for code in book.get("languages", []) if isinstance(code, str)][:3],
        # Retained so the resolver can promote it to a provenance-bearing
        # observation without spending a second request on the same book.
        discovery_payload=book,
    )


async def _collect(settings: Settings, max_candidates: int) -> list[CandidateBook]:
    base = settings.gutendex_base_url.rstrip("/")
    collected: list[CandidateBook] = []
    url: str | None = f"{base}/books/"

    async with httpx.AsyncClient(
        timeout=settings.http_read_timeout_seconds,
        # `next` is an absolute URL from the same host; following redirects
        # also covers the trailing-slash rule the index enforces.
        follow_redirects=True,
        headers={"User-Agent": settings.user_agent()},
    ) as client:
        while url and len(collected) < max_candidates:
            response = await client.get(url)
            response.raise_for_status()
            body = response.json()

            for book in body.get("results", []):
                candidate = _to_candidate(book)
                if candidate is not None:
                    collected.append(candidate)
                if len(collected) >= max_candidates:
                    break

            url = body.get("next")
            if url:
                # One request per second, the same courtesy the other adapters
                # extend. Gutendex is a small volunteer-run service.
                await asyncio.sleep(1.0)

    return collected


def build_manifest_from_gutendex(
    settings: Settings, manifest_path: Path, *, max_candidates: int
) -> int:
    """Write a candidate manifest from Gutendex.

    Returns the number written. The manifest format is identical to the dump's,
    so everything downstream is unchanged — which is the point of discovering
    into a manifest rather than straight into the resolver.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    candidates = asyncio.run(_collect(settings, max_candidates))

    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate.model_dump(mode="json")) + "\n")
    # Atomic: a crash mid-write must not leave a half-manifest that the next
    # task reads as complete.
    temporary.replace(manifest_path)

    logger.info(
        "discovery.gutendex_manifest_written",
        candidates=len(candidates),
        path=str(manifest_path),
    )
    return len(candidates)
