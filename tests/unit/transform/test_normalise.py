"""Normalisation.

These produce the comparison forms that identity and deduplication depend on,
so two spellings of the same thing must collapse and two different things must
not. Display values are always preserved separately.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pipeline.transform.normalise import (
    normalise_author,
    normalise_language,
    normalise_subject,
    normalise_title,
    parse_author_year,
    parse_year,
    select_language,
)


class TestNormaliseTitle:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Moby Dick  ", "moby dick"),
            ("Moby\t\nDick", "moby dick"),
            ("Moby   Dick", "moby dick"),
            ("MOBY DICK", "moby dick"),
        ],
    )
    def test_trims_collapses_and_casefolds(self, raw: str, expected: str) -> None:
        assert normalise_title(raw) == expected

    def test_unicode_forms_of_the_same_text_agree(self) -> None:
        # "é" as one codepoint and as "e" plus a combining accent look
        # identical and would otherwise be two different books.
        composed = "Les Misérables"
        decomposed = "Les Misérables"

        assert composed != decomposed
        assert normalise_title(composed) == normalise_title(decomposed)

    def test_series_text_is_kept(self) -> None:
        # "do not remove meaningful series text without a tested rule" — book 1
        # and book 2 of a series are different books with similar titles.
        assert "book two" in normalise_title("The Long Earth: Book Two")

    def test_punctuation_variants_collapse(self) -> None:
        # A typographic apostrophe and a straight one are the same title.
        assert normalise_title("Kelly's Heroes") == normalise_title("Kelly’s Heroes")

    def test_returns_none_for_blank_input(self) -> None:
        assert normalise_title("   ") is None
        assert normalise_title(None) is None

    @given(st.text())
    def test_is_idempotent(self, value: str) -> None:
        once = normalise_title(value)
        assert normalise_title(once) == once

    @given(st.text(min_size=1))
    def test_never_returns_an_empty_string(self, value: str) -> None:
        # An empty string would be a valid-looking identity component.
        assert normalise_title(value) != ""


class TestParseYear:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1997", 1997),
            ("c1997", 1997),
            ("c. 1997", 1997),
            ("[1997]", 1997),
            ("1997-03-01", 1997),
            ("1997/03/01", 1997),
            ("March 1997", 1997),
            ("©1997", 1997),
            ("1997.", 1997),
            ("MCMXCVII", 1997),
            ("mcmxcvii", 1997),
            ("MDCCCXCV", 1895),
        ],
    )
    def test_parses_the_forms_sources_actually_send(self, raw: str, expected: int) -> None:
        assert parse_year(raw) == expected

    @pytest.mark.parametrize("raw", ["1399", "2101", "0", "-500", "12345"])
    def test_rejects_years_outside_the_schema_constraint(self, raw: str) -> None:
        # published_year has CHECK (published_year BETWEEN 1400 AND 2100); a
        # value the database will refuse must never reach it.
        assert parse_year(raw) is None

    @pytest.mark.parametrize("raw", ["", "   ", "n.d.", "unknown", "forthcoming", None])
    def test_returns_none_for_unusable_input(self, raw: str | None) -> None:
        assert parse_year(raw) is None

    def test_prefers_a_four_digit_year_over_a_roman_numeral_fragment(self) -> None:
        # "MDCCCXCV (1895)" carries both; they agree, and the digits are safer.
        assert parse_year("MDCCCXCV (1895)") == 1895

    def test_an_out_of_range_roman_numeral_is_rejected(self) -> None:
        assert parse_year("MMMCC") is None

    @given(st.text(max_size=30))
    def test_never_raises_and_never_returns_an_illegal_year(self, value: str) -> None:
        year = parse_year(value)
        assert year is None or 1400 <= year <= 2100


class TestParseAuthorYear:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(-750, -750), ("-750", -750), (" 1817 ", 1817), (2100, 2100)],
    )
    def test_preserves_signed_years(self, raw: int | str, expected: int) -> None:
        assert parse_author_year(raw) == expected

    @pytest.mark.parametrize("raw", [True, "unknown", "-5000", 3000, "", None])
    def test_rejects_unusable_years(self, raw: int | str | None) -> None:
        assert parse_author_year(raw) is None


class TestNormaliseLanguage:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("en", "eng"),
            ("eng", "eng"),
            ("EN", "eng"),
            ("fr", "fra"),
            ("fre", "fra"),  # 639-2/B bibliographic to 639-3 terminological
            ("ger", "deu"),
            ("de", "deu"),
            ("cze", "ces"),
        ],
    )
    def test_maps_639_1_and_639_2_to_639_3(self, raw: str, expected: str) -> None:
        assert normalise_language(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "zz", "xyzzy", "123", None])
    def test_returns_none_for_unrecognised_codes(self, raw: str | None) -> None:
        assert normalise_language(raw) is None

    def test_output_always_matches_the_schema_constraint(self) -> None:
        # language has CHECK (language ~ '^[a-z]{3}$')
        for code in ("en", "fre", "ger", "spa", "ja"):
            result = normalise_language(code)
            assert result is not None
            assert len(result) == 3
            assert result.islower()


class TestSelectLanguage:
    def test_a_single_code_is_used(self) -> None:
        assert select_language(["en"]) == "eng"

    def test_repeats_of_one_language_still_resolve(self) -> None:
        assert select_language(["en", "eng", "EN"]) == "eng"

    def test_genuinely_multilingual_records_resolve_to_none(self) -> None:
        # Open Library lists one language per edition of a work. A live run
        # returned ["cze", ...] first for Dorian Gray. Recording Czech for an
        # English classic is worse than recording nothing, and there is no
        # information in the response to break the tie.
        assert select_language(["eng", "cze", "fre"]) is None

    def test_unrecognised_codes_are_ignored_rather_than_counted(self) -> None:
        assert select_language(["en", "zz", "!!"]) == "eng"

    @pytest.mark.parametrize("empty", [[], None])
    def test_handles_absent_input(self, empty: list[str] | None) -> None:
        assert select_language(empty) is None


class TestNormaliseAuthor:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Melville, Herman", "herman melville"),
            ("Herman Melville", "herman melville"),
            ("  Herman   Melville  ", "herman melville"),
            ("MELVILLE, HERMAN", "herman melville"),
        ],
    )
    def test_surname_first_and_natural_order_agree(self, raw: str, expected: str) -> None:
        # Gutendex writes "Melville, Herman"; Google Books writes the reverse.
        # Treating them as two people would split a canonical book in half.
        assert normalise_author(raw) == expected

    def test_trailing_life_dates_are_dropped(self) -> None:
        # Open Library sometimes appends them to the display name.
        assert normalise_author("Austen, Jane, 1775-1817") == "jane austen"

    def test_punctuation_and_accents_are_folded(self) -> None:
        assert normalise_author("Émile Zola") == normalise_author("Emile Zola")

    def test_initials_are_kept(self) -> None:
        # "J. R. R. Tolkien" and "John Tolkien" are not interchangeable.
        assert normalise_author("Tolkien, J. R. R.") == "j r r tolkien"

    def test_returns_none_for_blank_input(self) -> None:
        assert normalise_author("  ") is None
        assert normalise_author(None) is None

    @given(st.text())
    def test_is_idempotent(self, value: str) -> None:
        once = normalise_author(value)
        assert normalise_author(once) == once


class TestNormaliseSubject:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Adventure stories", "adventure stories"),
            ("  Science Fiction  ", "science fiction"),
            ("Whaling -- Fiction", "whaling -- fiction"),
        ],
    )
    def test_produces_a_stable_comparison_form(self, raw: str, expected: str) -> None:
        assert normalise_subject(raw) == expected

    def test_returns_none_for_blank_input(self) -> None:
        assert normalise_subject("") is None

    @given(st.text())
    def test_is_idempotent(self, value: str) -> None:
        once = normalise_subject(value)
        assert normalise_subject(once) == once
