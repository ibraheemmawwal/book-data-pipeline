"""Normalisation into stable comparison forms.

Identity and deduplication are built on these outputs, so the bar is that two
spellings of the same thing collapse and two different things never do. Display
values are preserved separately — normalisation is for matching, not for what a
reader sees.

Everything here is pure and total: no I/O, and no input raises.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pycountry

from pipeline.models.domain import AUTHOR_YEAR_MAX, AUTHOR_YEAR_MIN

# Mirrors CHECK (published_year BETWEEN 1400 AND 2100). A value the database
# would refuse must never reach it.
ALPHA_3_LENGTH = 3

MIN_YEAR = 1400
MAX_YEAR = 2100

_WHITESPACE = re.compile(r"\s+")
_FOUR_DIGIT_YEAR = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_ROMAN_NUMERAL = re.compile(r"(?<![A-Za-z])([MDCLXVImdclxvi]{2,})(?![A-Za-z])")
_LIFE_DATES = re.compile(r",?\s*\d{3,4}\s*[-\u2013\u2014]\s*\d{0,4}\s*$")
_AUTHOR_PUNCTUATION = re.compile(r"[.,;:()\[\]]")
_SIGNED_YEAR = re.compile(r"^[+-]?\d{1,4}$")

# Typographic characters that carry no distinguishing meaning in a title.
_PUNCTUATION_FOLD = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u00a0": " ",
    }
)

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_ROMAN_RENDER = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def _fold(value: str) -> str:
    """Casefold, collapse whitespace and normalise Unicode to NFC.

    NFC matters because "é" as one codepoint and as "e" plus a combining accent
    render identically but compare unequal — without this they are two books.
    """
    text = unicodedata.normalize("NFC", value).translate(_PUNCTUATION_FOLD)
    return _WHITESPACE.sub(" ", text).strip().casefold()


def normalise_title(value: str | None) -> str | None:
    """A title's comparison form.

    Series text is deliberately kept: "Book Two" is what distinguishes one
    volume from another, and stripping it would merge a series into one row.
    """
    if value is None:
        return None
    folded = _fold(value)
    return folded or None


def normalise_subject(value: str | None) -> str | None:
    """A subject heading's comparison form."""
    if value is None:
        return None
    folded = _fold(value)
    return folded or None


def normalise_author(value: str | None) -> str | None:
    """An author name's comparison form, in natural order.

    Gutendex writes "Melville, Herman" and Google Books writes "Herman
    Melville". Treating those as two people would split one canonical book
    across two rows, so surname-first is inverted to natural order.

    Initials are preserved: "J. R. R. Tolkien" and "John Tolkien" are not
    interchangeable, and collapsing them would merge distinct authors.
    """
    if value is None:
        return None

    text = unicodedata.normalize("NFC", value).translate(_PUNCTUATION_FOLD).strip()
    text = _LIFE_DATES.sub("", text)

    # Exactly one comma means surname-first; more than one is a corporate or
    # compound name we should not try to reorder.
    if text.count(",") == 1:
        surname, _, forenames = text.partition(",")
        if surname.strip() and forenames.strip():
            text = f"{forenames.strip()} {surname.strip()}"

    text = _AUTHOR_PUNCTUATION.sub(" ", text)
    # Strip accents so "Émile" and "Emile" are one author.
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    folded = _WHITESPACE.sub(" ", text).strip().casefold()
    return folded or None


def _int_to_roman(value: int) -> str:
    """Render canonically, so a parse can be checked by round trip."""
    parts = []
    for amount, numeral in _ROMAN_RENDER:
        count, value = divmod(value, amount)
        parts.append(numeral * count)
    return "".join(parts)


def _roman_to_int(value: str) -> int | None:
    """Convert a Roman numeral, rejecting anything not in canonical form.

    Matching case-insensitively means ordinary words made of Roman letters —
    "did", "dim", "mix" — reach this function. Requiring the value to render
    back to exactly what was read rejects them: "DID" parses arithmetically to
    999, but 999 renders as "CMXCIX", so it was never a numeral.
    """
    upper = value.upper()
    total = 0
    previous = 0
    for char in reversed(upper):
        current = _ROMAN_VALUES.get(char)
        if current is None:
            return None
        total = total - current if current < previous else total + current
        previous = max(previous, current)

    if total <= 0 or _int_to_roman(total) != upper:
        return None
    return total


def parse_year(value: str | None) -> int | None:
    """Extract a publication year from whatever the source sent.

    Sources send ``1997``, ``c1997``, ``1997-03-01``, ``[1997]``, ``©1997`` and
    occasionally ``MCMXCVII``. Digits are tried first: when a record carries
    both forms they agree, and digits cannot be confused with a word.

    Returns ``None`` rather than raising for anything unusable, and never
    returns a year the schema would reject.
    """
    if not value:
        return None

    text = unicodedata.normalize("NFC", value).strip()

    match = _FOUR_DIGIT_YEAR.search(text)
    if match is not None:
        year = int(match.group(1))
        return year if MIN_YEAR <= year <= MAX_YEAR else None

    roman = _ROMAN_NUMERAL.search(text)
    if roman is not None:
        year = _roman_to_int(roman.group(1)) or 0
        return year if MIN_YEAR <= year <= MAX_YEAR else None

    return None


def parse_author_year(value: int | str | None) -> int | None:
    """Validate a signed author year while retaining BCE values."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        year = value
    else:
        text = value.strip()
        if not _SIGNED_YEAR.fullmatch(text):
            return None
        year = int(text)

    return year if AUTHOR_YEAR_MIN <= year <= AUTHOR_YEAR_MAX else None


@lru_cache(maxsize=2048)
def normalise_language(value: str | None) -> str | None:
    """Map an ISO 639-1 or 639-2 code to 639-3.

    Sources disagree: Gutendex sends ``en``, Open Library sends ``eng``, and
    Open Library's bibliographic codes (``fre``, ``ger``) differ from the
    terminological ones (``fra``, ``deu``) the catalogue stores.

    Cached because a single run maps the same handful of codes thousands of
    times, and ``pycountry`` lookups are not free.
    """
    if not value:
        return None

    code = value.strip().lower()
    if not code.isalpha():
        return None

    for attribute in ("alpha_2", "alpha_3", "bibliographic"):
        try:
            language = pycountry.languages.get(**{attribute: code})
        except (LookupError, KeyError):
            language = None
        if language is not None:
            resolved: str | None = getattr(language, "alpha_3", None)
            if resolved is not None and len(resolved) == ALPHA_3_LENGTH:
                return resolved.lower()

    return None


def select_language(values: list[str] | None) -> str | None:
    """Choose one language for a record, or ``None`` if that would be a guess.

    Open Library lists one language per *edition* of a work, so a well-known
    title arrives with a dozen. A live run put ``cze`` first for *The Picture of
    Dorian Gray*; recording Czech for an English classic is worse than recording
    nothing, and the response carries no counts or ordering to break the tie
    with.

    A single distinct language — after normalisation, so ``en`` and ``eng``
    agree — is unambiguous and used. Anything else resolves to ``None``.
    """
    if not values:
        return None

    distinct = {
        normalised for value in values if (normalised := normalise_language(value)) is not None
    }
    return distinct.pop() if len(distinct) == 1 else None
