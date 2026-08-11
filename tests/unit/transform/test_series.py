"""Series identity and normalisation.

A series is a join key. Two spellings of the same series must collapse into one
row, and two genuinely different series must never merge — a wrong merge would
attach unrelated books to each other permanently, and nothing downstream could
detect it.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from typing import Any

import pytest

from pipeline.models.domain import RawSeriesMembership, SourceName
from pipeline.transform.series import (
    NAME_SERIES_PREFIX,
    SOURCE_SERIES_PREFIX,
    canonicalise_series,
    normalise_series,
    parse_series_position,
    series_identity_key,
    series_search_text,
)


class TestNormaliseSeries:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("A Song of Ice and Fire", "a song of ice and fire"),
            ("  Discworld  ", "discworld"),
            ("The\tLord   of the Rings", "the lord of the rings"),
        ],
    )
    def test_produces_a_stable_comparison_form(self, raw: str, expected: str) -> None:
        assert normalise_series(raw) == expected

    def test_html_entities_are_decoded_before_normalisation(self) -> None:
        # Encoded residue inside a join key would split one series in two.
        assert normalise_series("Tom &amp; Jerry") == normalise_series("Tom & Jerry")

    def test_unicode_forms_agree(self) -> None:
        assert normalise_series("Les Misérables") == normalise_series("Les Misérables")

    def test_typographic_punctuation_folds(self) -> None:
        assert normalise_series("Kelly’s Saga") == normalise_series("Kelly's Saga")

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_input_yields_none(self, blank: str | None) -> None:
        assert normalise_series(blank) is None


class TestParseSeriesPosition:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1", "1"), ("2.5", "2.5"), ("0.5", "0.5"), (" 7.5 ", "7.5"), ("10", "10")],
    )
    def test_parses_exact_decimals(self, raw: str, expected: str) -> None:
        assert parse_series_position(raw) == Decimal(expected)

    def test_positions_stay_exact_rather_than_binary(self) -> None:
        # 2.5 is a number a reader sees on the cover; float rounding would make
        # it compare unequal to the stored value.
        assert parse_series_position("2.5") == Decimal("2.5")

    @pytest.mark.parametrize("bad", ["", "  ", "one", "1.2.3", "-1", "abc", None])
    def test_unusable_positions_yield_none(self, bad: str | None) -> None:
        assert parse_series_position(bad) is None

    def test_an_absurd_position_is_refused(self) -> None:
        # NUMERIC(8,2) caps what the column can hold.
        assert parse_series_position("99999999") is None


class TestSeriesIdentityKey:
    def test_a_confirmed_source_id_wins(self) -> None:
        key = series_identity_key(
            source=SourceName.GOODREADS,
            source_series_id="45175",
            normalised_name="a song of ice and fire",
            confirmed=True,
        )

        assert key == f"{SOURCE_SERIES_PREFIX}goodreads:45175"

    def test_an_unconfirmed_source_id_is_not_used(self) -> None:
        # An id we could not verify against the /series/ slug is a guess, and a
        # guess must not become the join key everything else merges on.
        key = series_identity_key(
            source=SourceName.GOODREADS,
            source_series_id="45175",
            normalised_name="a song of ice and fire",
            confirmed=False,
        )

        assert key.startswith(NAME_SERIES_PREFIX)

    def test_without_a_source_id_it_falls_back_to_the_name(self) -> None:
        key = series_identity_key(
            source=SourceName.GOODREADS,
            source_series_id=None,
            normalised_name="discworld",
            confirmed=False,
        )

        assert key == NAME_SERIES_PREFIX + _digest_of("discworld")

    def test_the_name_key_is_deterministic(self) -> None:
        args: dict[str, Any] = {
            "source": SourceName.GOODREADS,
            "source_series_id": None,
            "normalised_name": "discworld",
            "confirmed": False,
        }

        assert series_identity_key(**args) == series_identity_key(**args)  # type: ignore[arg-type]

    def test_the_same_name_from_two_sources_collapses(self) -> None:
        # Name-based identity is source-independent by design; that is what
        # lets Open Library and Goodreads agree on one series row.
        left = series_identity_key(
            source=SourceName.GOODREADS,
            source_series_id=None,
            normalised_name="discworld",
            confirmed=False,
        )
        right = series_identity_key(
            source=SourceName.OPENLIBRARY,
            source_series_id=None,
            normalised_name="discworld",
            confirmed=False,
        )

        assert left == right

    def test_different_names_give_different_keys(self) -> None:
        a = series_identity_key(
            source=SourceName.GOODREADS,
            source_series_id=None,
            normalised_name="discworld",
            confirmed=False,
        )
        b = series_identity_key(
            source=SourceName.GOODREADS,
            source_series_id=None,
            normalised_name="dune",
            confirmed=False,
        )

        assert a != b

    def test_a_blank_name_without_a_source_id_is_refused(self) -> None:
        # Every unidentifiable series would otherwise collapse onto one row.
        with pytest.raises(ValueError, match="name"):
            series_identity_key(
                source=SourceName.GOODREADS,
                source_series_id=None,
                normalised_name="",
                confirmed=False,
            )


def _digest_of(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


class TestCanonicaliseSeries:
    def test_maps_a_raw_membership(self) -> None:
        clean = canonicalise_series(
            RawSeriesMembership(
                name="A Song of Ice and Fire",
                source_series_id="45175",
                position="1",
                confirmed=True,
            ),
            source=SourceName.GOODREADS,
        )

        assert clean is not None
        assert clean.name == "A Song of Ice and Fire"
        assert clean.normalised_name == "a song of ice and fire"
        assert clean.position == Decimal("1")
        assert clean.confirmed is True

    def test_the_display_name_is_decoded_but_preserved(self) -> None:
        clean = canonicalise_series(
            RawSeriesMembership(name="Tom &amp; Jerry"), source=SourceName.GOODREADS
        )

        assert clean is not None
        assert clean.name == "Tom & Jerry"
        assert clean.normalised_name == "tom & jerry"

    def test_an_unusable_position_does_not_lose_the_series(self) -> None:
        clean = canonicalise_series(
            RawSeriesMembership(name="Discworld", position="nonsense"),
            source=SourceName.GOODREADS,
        )

        assert clean is not None
        assert clean.position is None

    def test_a_name_that_normalises_away_is_dropped(self) -> None:
        # RawSeriesMembership already forbids a blank name, so this guard is
        # defence in depth: canonicalise_series must not assume its input came
        # through that validator. Constructed unvalidated to reach the branch.
        unvalidated = RawSeriesMembership.model_construct(
            name="   ", source_series_id=None, position=None, confirmed=False
        )

        assert canonicalise_series(unvalidated, source=SourceName.GOODREADS) is None


class TestSeriesSearchText:
    def test_joins_names_in_normalised_order(self) -> None:
        # Deterministic ordering: the loader recomputes this on every ingest,
        # and an order that wobbled would rewrite rows that had not changed.
        assert series_search_text(["Discworld", "A Song of Ice and Fire"]) == (
            "A Song of Ice and Fire Discworld"
        )

    def test_duplicates_collapse(self) -> None:
        assert series_search_text(["Discworld", "discworld"]) == "Discworld"

    def test_no_series_yields_an_empty_string(self) -> None:
        # The column is NOT NULL DEFAULT '', so None would violate the schema.
        assert series_search_text([]) == ""

    def test_blank_names_are_ignored(self) -> None:
        assert series_search_text(["  ", "Discworld"]) == "Discworld"
