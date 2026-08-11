"""ISBN validation and conversion.

The checksum is the whole point. An unvalidated ISBN becomes a canonical
identity key, and a wrong one silently merges two different books into one row
that no later stage can pull apart.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from pipeline.transform.isbn import (
    is_valid_isbn13,
    isbn10_check_digit,
    isbn13_check_digit,
    select_canonical_isbn,
    to_isbn13,
)


class TestToIsbn13:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("9780553380163", "9780553380163"),
            ("978-0-553-38016-3", "9780553380163"),
            ("978 0 553 38016 3", "9780553380163"),
            ("  9780553380163  ", "9780553380163"),
        ],
    )
    def test_accepts_a_valid_isbn13_in_any_common_formatting(
        self, value: str, expected: str
    ) -> None:
        assert to_isbn13(value) == expected

    @pytest.mark.parametrize(
        ("isbn10", "expected"),
        [
            ("0553380168", "9780553380163"),
            ("0-553-38016-8", "9780553380163"),
            ("080442957X", "9780804429573"),
            ("043942089X", "9780439420891"),
        ],
    )
    def test_converts_a_valid_isbn10(self, isbn10: str, expected: str) -> None:
        assert to_isbn13(isbn10) == expected

    def test_accepts_a_lowercase_x_check_digit(self) -> None:
        assert to_isbn13("080442957x") == "9780804429573"

    @pytest.mark.parametrize(
        "bad",
        [
            "9780553380164",  # ISBN-13 with a wrong check digit
            "0553380169",  # ISBN-10 with a wrong check digit
            "97805533801",  # too short
            "97805533801631",  # too long
            "abcdefghijklm",
            "",
            "   ",
            "X780553380163",  # X is only legal as an ISBN-10 check digit
        ],
    )
    def test_rejects_invalid_input(self, bad: str) -> None:
        assert to_isbn13(bad) is None

    def test_rejects_none(self) -> None:
        assert to_isbn13(None) is None

    @pytest.mark.parametrize("body", ["978047118651", "979847118651"])
    def test_accepts_registered_isbn_prefixes(self, body: str) -> None:
        candidate = body + str(isbn13_check_digit(body))

        assert to_isbn13(candidate) == candidate

    def test_rejects_the_979_0_ismn_prefix(self) -> None:
        body = "979047118651"
        candidate = body + str(isbn13_check_digit(body))

        assert to_isbn13(candidate) is None

    def test_rejects_a_thirteen_digit_number_with_an_unregistered_prefix(self) -> None:
        # Only 978 and 979 are ISBN prefixes; 977 is ISSN and 590 is nothing.
        body = "9770553380"
        assert to_isbn13(body + str(isbn13_check_digit(body)) + "00") is None


class TestCheckDigits:
    def test_isbn13_check_digit_matches_a_known_value(self) -> None:
        assert isbn13_check_digit("978055338016") == 3

    def test_isbn10_check_digit_matches_a_known_value(self) -> None:
        assert isbn10_check_digit("055338016") == "8"

    def test_isbn10_check_digit_can_be_x(self) -> None:
        assert isbn10_check_digit("080442957") == "X"

    def test_is_valid_isbn13_rejects_a_single_digit_change(self) -> None:
        assert is_valid_isbn13("9780553380163")
        assert not is_valid_isbn13("9780553380173")


class TestProperties:
    @given(st.integers(min_value=0, max_value=999_999_999))
    def test_every_generated_isbn10_round_trips_to_a_valid_isbn13(self, body: int) -> None:
        nine = f"{body:09d}"
        isbn10 = nine + isbn10_check_digit(nine)

        converted = to_isbn13(isbn10)

        assert converted is not None
        assert is_valid_isbn13(converted)
        # The 978 prefix plus the original nine digits, re-checksummed.
        assert converted[3:12] == nine

    @given(st.integers(min_value=0, max_value=999_999_999))
    def test_conversion_is_idempotent(self, body: int) -> None:
        nine = f"{body:09d}"
        once = to_isbn13(nine + isbn10_check_digit(nine))

        assert once is not None
        assert to_isbn13(once) == once

    @given(
        st.integers(min_value=0, max_value=999_999_999),
        st.integers(min_value=0, max_value=8),
        st.integers(min_value=1, max_value=9),
    )
    def test_corrupting_one_digit_is_always_detected(
        self, body: int, position: int, shift: int
    ) -> None:
        # This is what the checksum exists for, and what protects identity.
        nine = f"{body:09d}"
        valid = to_isbn13(nine + isbn10_check_digit(nine))
        assert valid is not None

        digits = list(valid)
        index = 3 + position
        digits[index] = str((int(digits[index]) + shift) % 10)
        corrupted = "".join(digits)
        assume(corrupted != valid)

        assert not is_valid_isbn13(corrupted)

    @given(st.text(max_size=20))
    def test_never_raises_on_arbitrary_text(self, value: str) -> None:
        result = to_isbn13(value)

        assert result is None or is_valid_isbn13(result)


class TestSelectCanonicalIsbn:
    def test_picks_the_only_valid_candidate(self) -> None:
        assert select_canonical_isbn(["not-an-isbn", "0553380168"]) == "9780553380163"

    def test_returns_none_when_nothing_is_valid(self) -> None:
        assert select_canonical_isbn(["", "123", "abc"]) is None

    def test_is_deterministic_regardless_of_input_order(self) -> None:
        candidates = ["9780441172719", "0553380168", "9780140449136"]

        first = select_canonical_isbn(candidates)
        second = select_canonical_isbn(list(reversed(candidates)))

        assert first == second
        assert first is not None

    def test_deduplicates_an_isbn10_and_its_isbn13_form(self) -> None:
        # The same book expressed both ways is one candidate, not two.
        assert select_canonical_isbn(["0553380168", "9780553380163"]) == "9780553380163"

    def test_refuses_to_choose_from_a_work_level_edition_list(self) -> None:
        # A live Open Library work returned 4,725 ISBNs across its editions.
        # Any single pick is one arbitrary edition standing in for the work, so
        # the record falls back to title-and-author identity instead.
        many = [
            body + str(isbn13_check_digit(body)) for body in (f"97805533{i:04d}" for i in range(50))
        ]
        assert all(is_valid_isbn13(i) for i in many)

        assert select_canonical_isbn(many) is None

    def test_the_work_level_threshold_is_on_distinct_books(self) -> None:
        # The same edition repeated in both ISBN forms must not count twice
        # towards "this looks like a whole work".
        pairs = ["0553380168", "9780553380163", "080442957X", "9780804429573"]

        assert select_canonical_isbn(pairs) is not None

    def test_a_small_edition_list_still_resolves(self) -> None:
        # Two or three ISBNs is an edition with a reissue, not a whole work.
        assert select_canonical_isbn(["0553380168", "9780441172719"]) is not None

    @pytest.mark.parametrize("empty", [[], None])
    def test_handles_absent_candidates(self, empty: list[str] | None) -> None:
        assert select_canonical_isbn(empty) is None
