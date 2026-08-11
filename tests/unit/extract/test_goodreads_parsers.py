"""Goodreads parsing helpers.

These read an undocumented contract, so they are written to fail closed: an
unrecognised shape yields None rather than a guess. Everything here is pure,
which is what makes an unofficial source testable without touching it.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from pipeline.extract.goodreads_parsers import (
    clean_html_text,
    is_placeholder_cover,
    parse_book_detail,
    parse_first_edition,
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

    def test_a_missing_leading_article_still_matches(self) -> None:
        # Three of four words shared; the candidate has one the query lacks.
        score = score_candidate("Game of Thrones", None, "A Game of Thrones", "Martin")

        assert score == pytest.approx(0.857, abs=0.01)

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


class TestParserEdgeCases:
    def test_an_unparseable_position_keeps_the_series(self) -> None:
        # The relationship is real even when the number is nonsense; dropping
        # the whole series over a bad position would lose more than it saves.
        parsed = parse_series_from_title("Book (Discworld, #1.2.3)")

        assert parsed is not None
        assert parsed.name == "Discworld"
        assert parsed.position is None

    def test_a_title_that_is_only_a_series_suffix_is_refused(self) -> None:
        # Nothing is left to be the book's own title.
        assert parse_series_from_title("(Discworld, #1)") is None

    def test_a_blank_series_name_is_refused(self) -> None:
        assert parse_series_from_title("Book (   )") is None

    def test_a_blank_candidate_title_scores_zero(self) -> None:
        assert score_candidate("Dune", None, "", "Herbert") == 0.0

    def test_html_with_only_markup_yields_none(self) -> None:
        assert clean_html_text("<div><span></span></div>") is None


DETAIL_FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


def markup(name: str) -> str:
    return (DETAIL_FIXTURES / name).read_text()


class TestParseBookDetail:
    def test_reads_page_count_from_json_ld(self) -> None:
        # JSON-LD first: it is structured data the site publishes deliberately,
        # so it changes less often than the markup around it.
        assert parse_book_detail(markup("goodreads_book_detail.html")).page_count == 835

    def test_falls_back_to_the_description_element(self) -> None:
        # The live JSON-LD carries no description, so the attribute fallback is
        # the only path to it.
        detail = parse_book_detail(markup("goodreads_book_detail.html"))

        assert detail.description is not None
        assert "preternatural event" in detail.description
        assert "<br>" not in detail.description

    def test_reads_the_series_from_the_aria_label(self) -> None:
        detail = parse_book_detail(markup("goodreads_book_detail.html"))

        assert detail.series is not None
        assert detail.series.name == "A Song of Ice and Fire"
        assert detail.series.position == "1"

    def test_a_page_with_no_series_link_is_unconfirmed(self) -> None:
        # The real page carries no /series/ href, so the relationship is
        # inferred rather than evidenced, and must say so.
        detail = parse_book_detail(markup("goodreads_book_detail.html"))

        assert detail.series is not None
        assert detail.series.confirmed is False
        assert detail.series.source_series_id is None

    def test_a_matching_series_link_confirms_the_relationship(self) -> None:
        html = (
            '<div aria-label="Book 1 in the Discworld series">'
            '<a href="/series/40650-discworld">Discworld</a></div>'
        )
        detail = parse_book_detail(html)

        assert detail.series is not None
        assert detail.series.confirmed is True
        assert detail.series.source_series_id == "40650"

    def test_the_raw_json_ld_is_retained(self) -> None:
        detail = parse_book_detail(markup("goodreads_book_detail.html"))

        assert detail.payload["json_ld"]["@type"] == "Book"

    @pytest.mark.parametrize(
        "html", ["", "<html></html>", "<html><body>nothing useful</body></html>"]
    )
    def test_an_unrecognisable_page_degrades_rather_than_raising(self, html: str) -> None:
        # An undocumented contract that changed shape must produce a thinner
        # observation, not fail the candidate.
        detail = parse_book_detail(html)

        assert detail.description is None
        assert detail.series is None

    def test_a_non_numeric_page_count_is_ignored(self) -> None:
        html = '<script type="application/ld+json">{"@type":"Book","numberOfPages":"lots"}</script>'

        assert parse_book_detail(html).page_count is None

    def test_a_page_count_supplied_as_a_string_is_accepted(self) -> None:
        html = '<script type="application/ld+json">{"@type":"Book","numberOfPages":"412"}</script>'

        assert parse_book_detail(html).page_count == 412

    @pytest.mark.parametrize("bad", ["0", "-5", "true"])
    def test_a_non_positive_page_count_is_refused(self, bad: str) -> None:
        html = (
            f'<script type="application/ld+json">{{"@type":"Book","numberOfPages":{bad}}}</script>'
        )

        assert parse_book_detail(html).page_count is None


class TestParseFirstEdition:
    def test_reads_only_the_first_edition_block(self) -> None:
        # The page lists every edition ever published; merging them would
        # invent a book that never existed.
        edition = parse_first_edition(markup("goodreads_work_editions.html"))

        assert edition.isbn13 == "9780553381689"
        assert edition.publisher == "Bantam Books"

    def test_an_isbn10_is_converted_through_checksum_validation(self) -> None:
        # The fixture's first block carries the ISBN-10 form.
        assert parse_first_edition(markup("goodreads_work_editions.html")).isbn13 == (
            "9780553381689"
        )

    def test_a_full_publication_date_is_read(self) -> None:
        assert parse_first_edition(markup("goodreads_work_editions.html")).published == (
            "August 4, 1997"
        )

    def test_a_month_only_date_is_accepted(self) -> None:
        html = '<div data-testid="editionCell">Published by X\nSeptember 1998</div>'

        assert parse_first_edition(html).published == "September 1998"

    def test_a_page_with_no_edition_block_yields_nothing(self) -> None:
        edition = parse_first_edition("<html><body>no editions</body></html>")

        assert edition.isbn13 is None
        assert edition.publisher is None

    def test_an_invalid_isbn_is_dropped_rather_than_stored(self) -> None:
        # A bad ISBN would become a canonical identity and merge wrong books.
        html = '<div data-testid="editionCell">ISBN 9780553381680</div>'

        assert parse_first_edition(html).isbn13 is None


class TestTitleMatching:
    """Scoring by words rather than characters.

    Edit distance cannot tell a book from its study guide: one title contains
    the other, so every containment rule scores it near perfect. That is how
    "Social Psychology" resolved to "Social Psychology: Study Guide" in a live
    run, and — because Goodreads wins title preference — rewrote the canonical
    title of a real book.
    """

    @pytest.mark.parametrize(
        ("query", "candidate"),
        [
            ("Social Psychology", "Social Psychology: Study Guide"),
            ("Dune", "Dune Messiah"),
            ("Dune", "Children of Dune"),
            ("A Game of Thrones", "A Game of Thrones: The Graphic Novel"),
        ],
    )
    def test_a_candidate_with_extra_words_is_refused(self, query: str, candidate: str) -> None:
        # Extra words in the candidate are what a different edition, a
        # companion volume or a sequel look like.
        assert score_candidate(query, None, candidate, None) < 0.4

    @pytest.mark.parametrize(
        ("query", "candidate"),
        [
            ("Social Psychology", "Social Psychology"),
            ("Game of Thrones", "A Game of Thrones"),
            ("Herbs and Spices", "Herbs & Spices"),
            ("The Hobbit", "Hobbit, The"),
        ],
    )
    def test_the_same_book_written_differently_still_matches(
        self, query: str, candidate: str
    ) -> None:
        assert score_candidate(query, None, candidate, None) >= 0.75

    def test_word_order_does_not_matter(self) -> None:
        # "Hobbit, The" scored 0.57 on edit distance and is the same book.
        assert score_candidate("The Hobbit", None, "Hobbit, The", None) == 1.0

    def test_punctuation_and_case_are_ignored(self) -> None:
        assert score_candidate("herbs & spices!", None, "Herbs and Spices", None) > 0

    def test_unrelated_titles_of_similar_length_score_near_zero(self) -> None:
        # The original failure mode: normalised Levenshtein put these at 0.41,
        # just over a 0.4 threshold.
        assert score_candidate("Quantum Chromodynamics", None, "A Game of Thrones", None) < 0.4


class TestAuthorCannotRescueATitle:
    def test_a_right_author_does_not_carry_a_wrong_title(self) -> None:
        # The exact live failure: same author, wrong book.
        assert (
            score_candidate("Social Psychology", "Baron", "Social Psychology: Study Guide", "Baron")
            == 0.0
        )

    def test_a_matching_author_still_improves_a_good_title(self) -> None:
        with_author = score_candidate("Dune", "Frank Herbert", "Dune", "Frank Herbert")
        wrong_author = score_candidate("Dune", "Someone Else", "Dune", "Frank Herbert")

        assert with_author > wrong_author

    def test_a_wrong_author_alone_does_not_reject_a_matching_title(self) -> None:
        # Providers spell names differently; a title match is the stronger
        # signal and should survive an author we cannot confirm.
        assert score_candidate("Dune", "F. Herbert", "Dune", "Frank Herbert") >= 0.4
