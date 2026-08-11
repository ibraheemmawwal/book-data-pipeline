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
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz

# Title weighs more than author: providers disagree far more about how to write
# a name than about what a book is called.
TITLE_WEIGHT = 0.6
AUTHOR_WEIGHT = 0.4

EXACT_SCORE = 1.0
_TITLE_AUTHOR_PARTS = 2
CONTAINMENT_SCORE = 0.9

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


def _similarity(left: str, right: str) -> float:
    """Exact, then containment, then normalised Levenshtein."""
    a, b = left.strip().casefold(), right.strip().casefold()
    if not a or not b:
        return 0.0
    if a == b:
        return EXACT_SCORE
    if a in b or b in a:
        return CONTAINMENT_SCORE
    return fuzz.ratio(a, b) / 100.0


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
    if not query_author or not candidate_author:
        return title_score

    author_score = _similarity(query_author, candidate_author)
    return TITLE_WEIGHT * title_score + AUTHOR_WEIGHT * author_score


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
            position = Decimal(match.group(2))
        except InvalidOperation:
            position = None
        else:
            if position < 0:
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
