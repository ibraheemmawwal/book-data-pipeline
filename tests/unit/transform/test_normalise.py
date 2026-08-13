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
    _roman_to_int,
    normalise_author,
    normalise_language,
    normalise_subject,
    normalise_title,
    parse_author_year,
    parse_year,
    select_language,
    strip_marc_subfields,
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


class TestNormaliseEdgeCases:
    @pytest.mark.parametrize("value", ["IIII", "VV", "IC", "XM"])
    def test_non_canonical_roman_numerals_are_refused(self, value: str) -> None:
        # IIII parses arithmetically to 4, but 4 renders as IV, so it was never
        # a numeral. The round trip is what keeps ordinary words out.
        assert parse_year(value) is None

    def test_a_roman_numeral_containing_a_non_roman_letter_is_refused(self) -> None:
        assert parse_year("MCMXCVIIZ") is None

    @pytest.mark.parametrize("code", ["qqq", "xx", "zzz"])
    def test_an_unassigned_language_code_returns_none(self, code: str) -> None:
        # pycountry raises rather than returning None for some inputs; a
        # missing language must never become an exception mid-batch.
        assert normalise_language(code) is None

    def test_a_language_code_of_the_wrong_length_returns_none(self) -> None:
        assert normalise_language("engl") is None


class TestRomanNumeralGuards:
    def test_a_numeral_containing_an_unknown_letter_is_refused(self) -> None:
        # The regex allows only Roman letters, so this exercises the guard that
        # protects _roman_to_int when called on anything else.
        assert _roman_to_int("MCMZ") is None

    def test_an_empty_numeral_is_refused(self) -> None:
        assert _roman_to_int("") is None


class TestMarcSubfieldDelimiters:
    """Markup from the MARC records behind Open Library's dump.

    "$b" introduces the remainder of a title and "$c" the statement of
    responsibility. They are not words, and a catalogue showing "Telling
    fortunes by cards : $b a symposium of..." is showing a reader its plumbing.
    """

    def test_a_subtitle_delimiter_is_removed(self) -> None:
        title = "Telling fortunes by cards : $b a symposium of the several ancient methods"

        assert strip_marc_subfields(title) == (
            "Telling fortunes by cards : a symposium of the several ancient methods"
        )

    def test_the_isbd_punctuation_is_left_alone(self) -> None:
        # "Title : subtitle" is how libraries write a title. Splitting it into
        # two fields is a different decision with an identity change behind it.
        assert strip_marc_subfields("Machine gun manual : $b a complete manual") == (
            "Machine gun manual : a complete manual"
        )

    def test_the_dagger_form_is_handled(self) -> None:
        assert strip_marc_subfields("Title ‡b with a dagger") == "Title with a dagger"

    @pytest.mark.parametrize(
        "title",
        [
            "A$AP Rocky greatest hits",
            "Priced at $5 and up",
            "Ke$ha: the story",
            "Money, money, money: the $ and the sense",
        ],
    )
    def test_real_dollar_signs_survive(self, title: str) -> None:
        """The failure mode a looser rule would introduce.

        A bare "$<letter>" rule mauls A$AP; a case-insensitive one is worse.
        MARC 21 specifies lowercase codes, so only those are matched, and only
        when followed by whitespace.
        """
        assert strip_marc_subfields(title) == title

    def test_a_plain_title_is_untouched(self) -> None:
        assert strip_marc_subfields("Dune") == "Dune"

    def test_none_stays_none(self) -> None:
        assert strip_marc_subfields(None) is None

    def test_a_title_of_only_markup_becomes_nothing(self) -> None:
        # Rather than an empty string, which would read as a real title.
        assert strip_marc_subfields("$b ") is None

    def test_a_title_without_markup_is_returned_untouched(self) -> None:
        """Not even whitespace is tidied.

        Display values are preserved verbatim; this exists to remove markup,
        not to become a general-purpose title cleaner that quietly rewrites
        every record that passes through it.
        """
        assert strip_marc_subfields("The   GREAT   Gatsby") == "The   GREAT   Gatsby"
