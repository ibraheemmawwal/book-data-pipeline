"""ISBN validation and conversion.

The checksum is the point. An ISBN becomes a canonical identity key, so an
unvalidated one silently merges two different books into a single row that no
later stage can pull apart — and unlike most data errors, this one is invisible
in the output.

Everything here is pure: same input, same answer, no I/O.
"""

from __future__ import annotations

import re

# Only 978 and 979 are ISBN prefixes. 977 is ISSN and 979-0 is ISMN for printed
# music; accepting either would let a non-book identifier become a book's
# identity.
VALID_PREFIXES = ("978", "979")
ISMN_PREFIX = "9790"

# Above this many candidates the record is describing a work rather than an
# edition. Open Library returned 4,725 ISBNs for one work in a live run, and
# any single pick is one arbitrary edition standing in for all of them — so the
# record falls back to title-and-author identity instead of asserting an ISBN
# it cannot justify.
MAX_CANDIDATES_FOR_IDENTITY = 8

_CHECK_DIGIT_X = 10

_SEPARATORS = re.compile(r"[\s\-\u2010-\u2015]")
_ISBN10 = re.compile(r"^[0-9]{9}[0-9X]$")
_ISBN13 = re.compile(r"^[0-9]{13}$")


def _clean(value: str) -> str:
    """Strip the hyphens, spaces and dashes publishers print ISBNs with."""
    return _SEPARATORS.sub("", value.strip()).upper()


def isbn13_check_digit(first_twelve: str) -> int:
    """The final digit of an ISBN-13, by the standard 1/3 alternating weights."""
    total = sum(
        int(digit) * (1 if index % 2 == 0 else 3) for index, digit in enumerate(first_twelve)
    )
    return (10 - total % 10) % 10


def isbn10_check_digit(first_nine: str) -> str:
    """The final character of an ISBN-10, which may legitimately be ``X``."""
    total = sum(int(digit) * (10 - index) for index, digit in enumerate(first_nine))
    remainder = (11 - total % 11) % 11
    return "X" if remainder == _CHECK_DIGIT_X else str(remainder)


def is_valid_isbn13(value: str) -> bool:
    """Whether ``value`` is thirteen digits, correctly prefixed and checksummed."""
    if not _ISBN13.fullmatch(value):
        return False
    if not value.startswith(VALID_PREFIXES):
        return False
    if value.startswith(ISMN_PREFIX):
        return False
    return int(value[12]) == isbn13_check_digit(value[:12])


def is_valid_isbn10(value: str) -> bool:
    """Whether ``value`` is a correctly checksummed ISBN-10."""
    if not _ISBN10.fullmatch(value):
        return False
    return value[9] == isbn10_check_digit(value[:9])


def to_isbn13(value: str | None) -> str | None:
    """Normalise any accepted ISBN form to a validated ISBN-13.

    Returns ``None`` for anything that fails validation rather than raising:
    a bad ISBN is ordinary source data, and the record around it is usually
    still worth keeping.
    """
    if not value:
        return None

    candidate = _clean(value)

    if is_valid_isbn13(candidate):
        return candidate

    if is_valid_isbn10(candidate):
        # ISBN-10 to ISBN-13 is a 978 prefix and a fresh check digit; the nine
        # significant digits are carried across unchanged.
        body = "978" + candidate[:9]
        return body + str(isbn13_check_digit(body))

    return None


def select_canonical_isbn(candidates: list[str] | None) -> str | None:
    """Choose one ISBN-13 to stand for a record, or ``None`` if that is a guess.

    Sources hand back anything from one ISBN to several thousand. A short list
    is an edition and its reissues, so picking deterministically is safe. A long
    list is a *work* — every edition ever published — where any single choice is
    arbitrary and would merge the work with whichever edition happened to sort
    first. Refusing is the honest answer; the caller falls back to
    title-and-author identity.

    Selection is by sorted order rather than input order so that the same
    record yields the same identity on every run, regardless of how the source
    happened to order its response.
    """
    if not candidates:
        return None

    # Deduplicate after conversion: an ISBN-10 and its ISBN-13 form are one
    # book, and counting them twice would inflate the work-level threshold.
    valid = {converted for c in candidates if (converted := to_isbn13(c)) is not None}

    if not valid or len(valid) > MAX_CANDIDATES_FOR_IDENTITY:
        return None

    return min(valid)
