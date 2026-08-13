"""Pure parsing helpers for the Goodreads adapter.

Goodreads ended public API access in 2020, so everything the adapter reads is
an undocumented web contract that can change without notice. These helpers are
therefore written to **fail closed**: an unrecognised shape returns ``None``
rather than a guess, because a wrong series name or a placeholder image stored
as a cover is worse than a missing field.

Keeping the parsing pure and separate from the HTTP adapter is what makes an
unofficial source testable at all — every rule below is exercised against
captured markup without touching the site.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from rapidfuzz import fuzz
from selectolax.parser import HTMLParser

from pipeline.models.domain import RawSeriesMembership
from pipeline.transform.isbn import to_isbn13

# Title weighs more than author: providers disagree far more about how to write
# a name than about what a book is called.
TITLE_WEIGHT = 0.6
AUTHOR_WEIGHT = 0.4

EXACT_SCORE = 1.0
_TITLE_AUTHOR_PARTS = 2

# A floor the title must clear on its own. The author confirms a match, it does
# not substitute for one: without this, a correct author would carry a wrong
# title over the line, which is exactly how "Social Psychology" resolved to the
# Study Guide edition in a live run and rewrote the canonical title.
MIN_TITLE_SIMILARITY = 0.75

# A much looser floor for ISBN lookups. An ISBN is an exact identifier, so
# Goodreads' own answer is better evidence than string similarity against a
# title we may have wrong — that is why ISBN queries skip ranking. But the two
# providers can simply disagree about which book an ISBN denotes, and when the
# answer shares almost nothing with what was asked for, one of them is wrong
# and guessing which is worse than falling back to a documented source.
ISBN_SANITY_FLOOR = 0.3

# Words shared with the query but absent from it are the signal that matters.
# Measured against real cases, this separates wanted matches (>= 0.80) from
# unwanted ones (<= 0.73) where character similarity could not: "Dune" scores
# 0.90 against "Dune Messiah" on edit distance and 0.67 on tokens.
_NO_SHARED_TOKEN_DISCOUNT = 0.5

_TOKEN_SPLIT = re.compile(r"[^0-9a-z]+")

_BY_SEPARATOR = re.compile(r"\s+by\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TAG = re.compile(r"<[^>]+>")
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_END = re.compile(r"</(p|div|li|h[1-6])>", re.IGNORECASE)

# "End of one.Start of two" — stripping tags routinely welds sentences.
_WELDED_SENTENCE = re.compile(r"([.!?])([A-Z])")

# Goodreads encodes the requested render size in the filename: 13496._SY75_.jpg
_COVER_SIZE_SEGMENT = re.compile(r"\._S[XY]\d+_(?=\.[a-z]{3,4}$)", re.IGNORECASE)

# A trailing "(Series Name, #2.5)" or "(Series Name)".
_SERIES_SUFFIX = re.compile(r"\s*\(([^()]+?)(?:,\s*#([\d.]+))?\)\s*$")

# Bracketed notes that look like a series but are not. Attaching books to an
# invented "Illustrated Edition" series would be worse than parsing nothing.
_NOT_A_SERIES = re.compile(
    r"^(?:\d{4}|"
    r".*\b(?:edition|illustrated|abridged|unabridged|reprint|paperback|"
    r"hardcover|boxed set|omnibus|annotated|translated|audiobook)\b.*)$",
    re.IGNORECASE,
)

_PLACEHOLDER_MARKERS = ("nophoto", "no-photo", "no_photo")

# Edition-block scraping. Deliberately narrow: a loose pattern that matched the
# wrong line would attribute one edition's ISBN to another.
_ISBN_IN_TEXT = re.compile(r"\b(?:97[89][\d-]{10,}|\d{9}[\dXx])\b")
_EDITION_DATE = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(?:\d{1,2},\s*)?\d{4}"
)
_EDITION_PUBLISHER = re.compile(r"(?:Published\s+by|Publisher:)\s*(.+)")


@dataclass(frozen=True, slots=True)
class ParsedSeries:
    """A series relationship recovered from a dirty title."""

    name: str
    position: Decimal | None
    bare_title: str


def split_title_by_author(query: str) -> tuple[str, str | None]:
    """Split ``Title by Author`` into its parts.

    Splits on the last separator, so "Life by Life by Kate Atkinson" keeps the
    repetition in the title. The word-boundary requirement stops "Goodbye to
    Berlin" being torn in half.
    """
    parts = _BY_SEPARATOR.split(query.strip())
    if len(parts) < _TITLE_AUTHOR_PARTS:
        return query.strip(), None
    return " by ".join(parts[:-1]).strip(), parts[-1].strip()


def _tokens(value: str) -> set[str]:
    return {token for token in _TOKEN_SPLIT.split(value.casefold()) if token}


def _similarity(left: str, right: str) -> float:
    """How much two titles say the same thing, by words rather than characters.

    Character edit distance cannot tell "Dune" from "Dune Messiah" — one
    contains the other, so every containment rule scores it near perfect. Word
    overlap can: the F1 of how much of the query the candidate covers against
    how much of the candidate the query accounts for. Extra words in the
    candidate are what a wrong edition or a study guide looks like, and they
    cost precision.

    Word order stops mattering, which is a bonus: "Hobbit, The" and "The
    Hobbit" are the same book and scored 0.57 on edit distance.
    """
    a, b = left.strip(), right.strip()
    if not a or not b:
        return 0.0

    query_tokens, candidate_tokens = _tokens(a), _tokens(b)
    if not query_tokens or not candidate_tokens:
        return 0.0
    if query_tokens == candidate_tokens:
        return EXACT_SCORE

    shared = query_tokens & candidate_tokens
    if not shared:
        # Nothing in common at word level. Keep a discounted character score so
        # a spelling difference is not treated as a different book, but never
        # let it reach the threshold on its own.
        return fuzz.ratio(a.casefold(), b.casefold()) / 100.0 * _NO_SHARED_TOKEN_DISCOUNT

    recall = len(shared) / len(query_tokens)
    precision = len(shared) / len(candidate_tokens)
    return 2 * recall * precision / (recall + precision)


def score_candidate(
    query_title: str,
    query_author: str | None,
    candidate_title: str,
    candidate_author: str | None,
) -> float:
    """How well an autocomplete result matches what was asked for.

    Only used for title/author queries. ISBN queries bypass scoring entirely —
    an ISBN is an exact identifier and Goodreads' own ordering is better
    evidence than string similarity against a title we may have wrong.
    """
    if not query_title.strip():
        return 0.0

    title_score = _similarity(query_title, candidate_title)
    if title_score < MIN_TITLE_SIMILARITY:
        # The author cannot rescue a wrong title. A study guide by the right
        # author is still not the book that was asked for.
        return 0.0

    if not query_author or not candidate_author:
        return title_score

    author_score = _similarity(query_author, candidate_author)
    return TITLE_WEIGHT * title_score + AUTHOR_WEIGHT * author_score


def is_plausible_isbn_match(query_title: str, candidate_title: str) -> bool:
    """Whether an ISBN lookup returned something recognisably the same book.

    Deliberately not the ranking threshold. This rejects a gross mismatch — a
    completely different book behind the same number — and nothing subtler. It
    will not catch a provider that maps an ISBN to a companion volume with a
    similar title, because that is indistinguishable from a provider that
    simply holds a fuller subtitle than we do.
    """
    if not query_title.strip() or not candidate_title.strip():
        return True
    return _similarity(query_title, candidate_title) >= ISBN_SANITY_FLOOR


def parse_series_from_title(dirty_title: str | None) -> ParsedSeries | None:
    """Recover a series from a Goodreads "dirty" title.

    Goodreads writes the series into the title as
    ``A Game of Thrones (A Song of Ice and Fire, #1)``. Positions are decimal
    because novellas really are numbered 0.5 and 2.5, and a float would not
    compare equal to the value printed on the book.

    Returns ``None`` for bracketed text that is an edition note or a year
    rather than a series.
    """
    if not dirty_title:
        return None

    match = _SERIES_SUFFIX.search(dirty_title)
    if match is None:
        return None

    name = match.group(1).strip()
    if not name or _NOT_A_SERIES.match(name):
        return None

    position: Decimal | None = None
    if match.group(2):
        try:
            # The capture group is digits and dots only, so a negative value
            # cannot reach here; a bare "." or "1.2.3" still can.
            position = Decimal(match.group(2))
        except InvalidOperation:
            position = None

    bare = dirty_title[: match.start()].strip()
    if not bare:
        return None

    return ParsedSeries(name=name, position=position, bare_title=bare)


def clean_html_text(value: str | None) -> str | None:
    """Turn a fragment of Goodreads markup into storable text.

    Order matters: breaks become newlines before tags are stripped, entities
    are decoded after stripping so an encoded ``&lt;b&gt;`` is not mistaken for
    markup, and welded sentences are repaired last.
    """
    if not value:
        return None

    text = _BREAK.sub("\n", value)
    text = _BLOCK_END.sub("\n\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    text = _WELDED_SENTENCE.sub(r"\1 \2", text)
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text).strip()
    return text or None


def upgrade_cover_url(url: str | None) -> str | None:
    """Drop the thumbnail size segment and force HTTPS.

    A URL-quality transform only. The bytes are never fetched or rehosted;
    that would need a separate content-rights decision.
    """
    if not url:
        return None
    cleaned = _COVER_SIZE_SEGMENT.sub("", url.strip())
    if cleaned.startswith("http://"):
        cleaned = "https://" + cleaned[len("http://") :]
    return cleaned or None


def is_placeholder_cover(url: str | None) -> bool:
    """Whether this is Goodreads' grey "no photo" image.

    Storing it as a cover would have the API serve a placeholder as though it
    were artwork, which is worse than returning no cover at all.
    """
    if not url:
        return True
    return any(marker in url.lower() for marker in _PLACEHOLDER_MARKERS)


_NON_SLUG = re.compile(r"[^a-z0-9]+")


_SERIES_HREF = re.compile(r"/series/(\d+)(?:-([a-z0-9-]+))?", re.IGNORECASE)

_ARIA_SERIES = re.compile(r"Book\s+([\d.]+)\s+in\s+the\s+(.+?)\s+series", re.IGNORECASE)


def _slugify(value: str) -> str:
    return _NON_SLUG.sub("-", value.strip().casefold()).strip("-")


def parse_series_id(href: str | None, series_name: str) -> str | None:
    """Accept a Goodreads series id only when its slug matches the name.

    An id taken from an unrelated link would attach a book to the wrong series
    permanently, and nothing downstream could detect it. When the slug does not
    agree the relationship is still recorded — just unconfirmed.
    """
    if not href:
        return None
    match = _SERIES_HREF.search(href)
    if match is None:
        return None
    slug = match.group(2)
    if slug and _slugify(slug) != _slugify(series_name):
        return None
    return match.group(1)


def parse_aria_series(label: str | None) -> tuple[str, Decimal | None] | None:
    """Read ``Book 2.5 in the Discworld series`` from an ARIA label."""
    if not label:
        return None
    match = _ARIA_SERIES.search(label)
    if match is None:
        return None
    try:
        position: Decimal | None = Decimal(match.group(1))
    except InvalidOperation:
        position = None
    name = match.group(2).strip()
    return (name, position) if name else None


# The only place a book page states its work id, and the work id is what the
# editions page is keyed on. An export that carries a book id but no work id
# would otherwise be unable to reach the editions at all.
_WORK_EDITIONS_LINK = re.compile(r"/work/editions/(\d+)")

# Goodreads publishes the date as epoch milliseconds in the page's own state,
# not in JSON-LD, which carries no date field of any kind.
_PUBLICATION_TIME = re.compile(r'"publicationTime"\s*:\s*(\d{10,})')


def _work_id(markup: str) -> str | None:
    """The work id a book page links to, if it links to one."""
    match = _WORK_EDITIONS_LINK.search(markup)
    return match.group(1) if match else None


def _publication_year(document: dict[str, Any]) -> str | None:
    """A publication date from JSON-LD, which usually carries none."""
    for key in ("datePublished", "dateCreated"):
        text = _as_text(document.get(key))
        if text:
            return text
    return None


def _published_from_page_state(markup: str) -> str | None:
    """The publication year, from epoch milliseconds in the page's own state.

    This is the only date a book page gives up. JSON-LD has no date field at
    all, and the editions page - which does - is keyed on a work id an export
    of book ids does not carry. Reading it here is what lets a bare Goodreads
    id produce a year.

    Returned as a year string because transform parses years, and a
    millisecond timestamp implies a precision the source does not have: the
    same book shows different times in different editions.
    """
    match = _PUBLICATION_TIME.search(markup)
    if match is None:
        return None
    try:
        moment = datetime.fromtimestamp(int(match.group(1)) / 1000, tz=UTC)
    except (ValueError, OSError, OverflowError):
        return None
    return str(moment.year)


def _json_ld_authors(document: dict[str, Any]) -> tuple[str, ...]:
    """Every author the page credits, in order.

    The search card carries a single ``author`` object, so a book with three
    contributors arrived with one and the other two were simply absent from the
    catalogue. JSON-LD carries the full array, which is the only place on the
    page they all appear.
    """
    raw = document.get("author")
    entries = raw if isinstance(raw, list) else [raw]

    names: list[str] = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else entry
        text = _as_text(name)
        if text and text not in names:
            names.append(text.strip())
    return tuple(names)


def parse_json_ld(html: str) -> dict[str, Any] | None:
    """Pull the Book JSON-LD block out of a detail page.

    Preferred over scraping attributes because it is structured data the site
    publishes deliberately, so it changes less often than the markup around it.
    """
    tree = HTMLParser(html)
    for node in tree.css('script[type="application/ld+json"]'):
        raw = node.text(strip=True)
        if not raw:
            continue
        try:
            document = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = document if isinstance(document, list) else [document]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") in {"Book", "Product"}:
                return candidate
    return None


def json_ld_authors(value: Any) -> list[str]:
    """JSON-LD ``author`` is an object or an array of them, never reliably one."""
    entries = value if isinstance(value, list) else [value]
    names = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("name"):
            names.append(str(entry["name"]).strip())
        elif isinstance(entry, str) and entry.strip():
            names.append(entry.strip())
    return names


@dataclass(frozen=True, slots=True)
class BookDetail:
    """What a ``/book/show/`` page adds to an autocomplete observation."""

    description: str | None = None
    page_count: int | None = None
    series: RawSeriesMembership | None = None
    authors: tuple[str, ...] = ()
    isbn: str | None = None
    published: str | None = None
    work_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EditionDetail:
    """What the first block of a ``/work/editions/`` page adds."""

    isbn13: str | None = None
    published: str | None = None
    publisher: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def parse_book_detail(markup: str) -> BookDetail:
    """Read a book detail page.

    JSON-LD first because it is structured data the site publishes
    deliberately, so it changes less often than the markup around it. ARIA
    labels are the fallback for series, which JSON-LD does not carry.

    Never raises. An undocumented contract that changed shape must degrade to a
    thinner observation, not fail the candidate.
    """
    payload: dict[str, Any] = {}
    description: str | None = None
    page_count: int | None = None
    authors: tuple[str, ...] = ()
    isbn: str | None = None
    published: str | None = None

    document = parse_json_ld(markup)
    if document is not None:
        payload["json_ld"] = document
        description = clean_html_text(_as_text(document.get("description")))
        page_count = _as_positive_int(document.get("numberOfPages"))
        authors = _json_ld_authors(document)
        isbn = _as_text(document.get("isbn"))
        published = _publication_year(document)

    tree = HTMLParser(markup)

    if description is None:
        node = tree.css_first('[data-testid="description"]')
        if node is not None:
            description = clean_html_text(node.html)

    series: RawSeriesMembership | None = None
    for node in tree.css("[aria-label]"):
        parsed = parse_aria_series(node.attributes.get("aria-label"))
        if parsed is None:
            continue
        name, position = parsed
        href = node.attributes.get("href")
        if href is None:
            link = node.css_first('a[href*="/series/"]')
            href = link.attributes.get("href") if link is not None else None
        source_series_id = parse_series_id(href, name)
        series = RawSeriesMembership(
            name=name,
            source_series_id=source_series_id,
            position=str(position) if position is not None else None,
            # Confirmed only when a /series/ link's slug agreed with the name.
            confirmed=source_series_id is not None,
        )
        payload["series_label"] = node.attributes.get("aria-label")
        payload["series_id"] = source_series_id
        break

    if published is None:
        published = _published_from_page_state(markup)

    work_id = _work_id(markup)
    if work_id:
        payload["work_id"] = work_id
    if isbn:
        payload["isbn"] = isbn
    if published:
        payload["published"] = published

    return BookDetail(
        description=description,
        page_count=page_count,
        series=series,
        authors=authors,
        isbn=isbn,
        published=published,
        work_id=work_id,
        payload=payload,
    )


def parse_first_edition(markup: str) -> EditionDetail:
    """Read only the first block of a work-editions page.

    One block, not all of them: the page lists every edition ever published,
    and merging them would invent a book that never existed.
    """
    tree = HTMLParser(markup)
    block = tree.css_first('[data-testid="editionCell"]') or tree.css_first(".editionData")
    if block is None:
        return EditionDetail()

    text = block.text(separator="\n")
    payload: dict[str, Any] = {"edition_text": text[:2000]}

    isbn13: str | None = None
    match = _ISBN_IN_TEXT.search(text)
    if match is not None:
        # Preserve a valid ISBN-10 check digit X and convert through checksum
        # validation rather than string surgery.
        isbn13 = to_isbn13(match.group(0))

    published: str | None = None
    date_match = _EDITION_DATE.search(text)
    if date_match is not None:
        published = date_match.group(0).strip()

    publisher: str | None = None
    publisher_match = _EDITION_PUBLISHER.search(text)
    if publisher_match is not None:
        publisher = publisher_match.group(1).strip() or None

    return EditionDetail(isbn13=isbn13, published=published, publisher=publisher, payload=payload)


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None
