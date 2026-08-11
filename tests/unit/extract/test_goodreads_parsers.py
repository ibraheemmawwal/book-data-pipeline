"""Goodreads parsing helpers.

These read an undocumented contract, so they are written to fail closed: an
unrecognised shape yields None rather than a guess. Everything here is pure,
which is what makes an unofficial source testable without touching it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from pipeline.extract.goodreads_parsers import (
    clean_html_text,
    is_placeholder_cover,
    parse_series_from_title,
    score_candidate,
    split_title_by_author,
    upgrade_cover_url,
)


class TestSplitTitleByAuthor:
    @pytest.mark.parametrize(
        ("query", "title", "author"),
        [
            ("A Game of Thrones by George R.R. Martin", "A Game of Thrones", "George R.R. Martin"),
            ("Dune by Frank Herbert", "Dune", "Frank Herbert"),
            ("Moby Dick", "Moby Dick", None),
        ],
    )
    def test_splits_on_the_by_separator(self, query: str, title: str, author: str | None) -> None:
        assert split_title_by_author(query) == (title, author)

    def test_a_title_containing_by_is_not_split_mid_word(self) -> None:
        # "Goodbye" contains "by"; splitting on the substring would mangle it.
        assert split_title_by_author("Goodbye to Berlin") == ("Goodbye to Berlin", None)

    def test_only_the_last_by_separates(self) -> None:
        title, author = split_title_by_author("Life by Life by Kate Atkinson")

        assert author == "Kate Atkinson"
        assert title == "Life by Life"


class TestScoreCandidate:
    def test_an_exact_title_match_scores_one(self) -> None:
        assert score_candidate("Dune", None, "Dune", "Frank Herbert") == 1.0

    def test_matching_is_case_insensitive(self) -> None:
        assert score_candidate("dune", None, "Dune", "Frank Herbert") == 1.0

    def test_substring_containment_scores_point_nine(self) -> None:
        score = score_candidate("Game of Thrones", None, "A Game of Thrones", "Martin")

        assert score == pytest.approx(0.9)

    def test_title_and_author_are_weighted_sixty_forty(self) -> None:
        # Exact title, exact author.
        assert score_candidate("Dune", "Frank Herbert", "Dune", "Frank Herbert") == 1.0
        # Exact title, wrong author: 0.6 of the weight survives.
        mixed = score_candidate("Dune", "Someone Else", "Dune", "Frank Herbert")
        assert 0.6 <= mixed < 1.0

    def test_an_unrelated_title_scores_below_the_threshold(self) -> None:
        assert score_candidate("Dune", None, "Pride and Prejudice", "Austen") < 0.4

    def test_a_missing_candidate_author_does_not_crash(self) -> None:
        assert score_candidate("Dune", "Frank Herbert", "Dune", None) > 0

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_query_scores_zero(self, blank: str) -> None:
        assert score_candidate(blank, None, "Dune", "Herbert") == 0.0


class TestParseSeriesFromTitle:
    @pytest.mark.parametrize(
        ("dirty", "name", "position"),
        [
            ("A Game of Thrones (A Song of Ice and Fire, #1)", "A Song of Ice and Fire", "1"),
            (
                "The Fellowship of the Ring (The Lord of the Rings, #1)",
                "The Lord of the Rings",
                "1",
            ),
            ("Tales (Discworld, #2.5)", "Discworld", "2.5"),
            ("Prequel (Foundation, #0.5)", "Foundation", "0.5"),
        ],
    )
    def test_extracts_name_and_position(self, dirty: str, name: str, position: str) -> None:
        parsed = parse_series_from_title(dirty)

        assert parsed is not None
        assert parsed.name == name
        assert parsed.position == Decimal(position)

    def test_decimal_positions_are_exact(self) -> None:
        # 2.5 as a float would not compare equal to the value a reader sees.
        parsed = parse_series_from_title("X (Y, #2.5)")

        assert parsed is not None
        assert parsed.position == Decimal("2.5")
        assert isinstance(parsed.position, Decimal)

    def test_a_series_without_a_position_still_parses(self) -> None:
        parsed = parse_series_from_title("Some Book (Some Series)")

        assert parsed is not None
        assert parsed.name == "Some Series"
        assert parsed.position is None

    @pytest.mark.parametrize(
        "title",
        ["A Game of Thrones", "Book (Illustrated Edition)", "Book (1998)", ""],
    )
    def test_returns_none_when_there_is_no_series(self, title: str) -> None:
        # An edition note or a year in brackets is not a series, and inventing
        # one would attach unrelated books to each other.
        assert parse_series_from_title(title) is None

    def test_the_bare_title_is_returned_alongside(self) -> None:
        parsed = parse_series_from_title("A Game of Thrones (A Song of Ice and Fire, #1)")

        assert parsed is not None
        assert parsed.bare_title == "A Game of Thrones"


class TestCleanHtmlText:
    def test_br_becomes_a_newline(self) -> None:
        assert clean_html_text("one<br>two") == "one\ntwo"
        assert clean_html_text("one<br/>two") == "one\ntwo"

    def test_markup_is_stripped(self) -> None:
        assert clean_html_text("<b>bold</b> text") == "bold text"

    def test_entities_are_decoded(self) -> None:
        assert clean_html_text("Tom &amp; Jerry &mdash; a tale") == "Tom & Jerry — a tale"

    def test_missing_punctuation_spacing_is_repaired(self) -> None:
        # Stripping tags routinely welds a sentence to the next one.
        assert clean_html_text("End of one.Start of two") == "End of one. Start of two"

    def test_whitespace_is_collapsed_without_losing_paragraphs(self) -> None:
        assert clean_html_text("a   b<br><br>c") == "a b\n\nc"

    @pytest.mark.parametrize("empty", ["", "   ", "<p></p>", None])
    def test_empty_input_yields_none(self, empty: str | None) -> None:
        assert clean_html_text(empty) is None


class TestCoverUrls:
    def test_thumbnail_size_segment_is_stripped(self) -> None:
        thumb = "https://i.gr-assets.com/images/S/photo/books/1562726234i/13496._SY75_.jpg"

        assert upgrade_cover_url(thumb) == (
            "https://i.gr-assets.com/images/S/photo/books/1562726234i/13496.jpg"
        )

    @pytest.mark.parametrize("segment", ["._SY75_", "._SX98_", "._SY475_"])
    def test_every_size_segment_form_is_stripped(self, segment: str) -> None:
        url = f"https://i.gr-assets.com/books/1i/9{segment}.jpg"

        assert upgrade_cover_url(url) == "https://i.gr-assets.com/books/1i/9.jpg"

    def test_a_url_without_a_size_segment_is_unchanged(self) -> None:
        url = "https://i.gr-assets.com/books/1i/9.jpg"

        assert upgrade_cover_url(url) == url

    def test_http_is_upgraded_to_https(self) -> None:
        assert upgrade_cover_url("http://i.gr-assets.com/x.jpg").startswith("https://")

    @pytest.mark.parametrize(
        "placeholder",
        [
            "https://s.gr-assets.com/assets/nophoto/book/111x148-bcc042a9c91a29c1d680899eff700a03.png",
            "https://www.goodreads.com/assets/nophoto/book/50x75.png",
        ],
    )
    def test_the_no_photo_placeholder_is_detected(self, placeholder: str) -> None:
        # Storing a placeholder as a cover is worse than storing nothing: the
        # API would serve a grey box as though it were artwork.
        assert is_placeholder_cover(placeholder)

    def test_a_real_cover_is_not_a_placeholder(self) -> None:
        assert not is_placeholder_cover("https://i.gr-assets.com/books/1i/13496.jpg")

    def test_none_is_handled(self) -> None:
        assert upgrade_cover_url(None) is None
        assert is_placeholder_cover(None)
