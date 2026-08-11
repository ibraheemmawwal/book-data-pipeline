"""Domain model boundary behaviour.

These models are the validation boundary. They must reject bad source data
loudly rather than letting it reach the transform or load stages.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pipeline.models.domain import CleanBook, RawAuthor, RawBook, SourceName


def raw_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "source": SourceName.GUTENDEX,
        "source_id": "1342",
        "title": "Pride and Prejudice",
        "raw_payload": {"id": 1342},
    }
    return base | overrides


class TestRawBook:
    def test_minimal_record_is_valid(self) -> None:
        book = RawBook(**raw_kwargs())  # type: ignore[arg-type]

        assert book.source is SourceName.GUTENDEX
        assert book.source_id == "1342"
        assert book.authors == []
        assert book.subjects == []
        assert book.source_updated is None

    def test_source_id_is_coerced_from_int(self) -> None:
        # Gutendex returns integer ids; Open Library returns strings.
        book = RawBook(**raw_kwargs(source_id=1342))  # type: ignore[arg-type]

        assert book.source_id == "1342"

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_blank_source_id_is_rejected(self, blank: str) -> None:
        with pytest.raises(ValidationError, match="source_id"):
            RawBook(**raw_kwargs(source_id=blank))  # type: ignore[arg-type]

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_title_is_rejected(self, blank: str) -> None:
        with pytest.raises(ValidationError, match="title"):
            RawBook(**raw_kwargs(title=blank))  # type: ignore[arg-type]

    def test_unknown_source_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source"):
            RawBook(**raw_kwargs(source="scraped-html"))  # type: ignore[arg-type]

    def test_unknown_field_is_rejected(self) -> None:
        # A silently ignored field is a silently dropped mapping bug.
        with pytest.raises(ValidationError, match="unexpected_field"):
            RawBook(**raw_kwargs(unexpected_field="x"))  # type: ignore[arg-type]

    def test_naive_source_updated_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RawBook(**raw_kwargs(source_updated=datetime(2026, 8, 11, 10, 0)))  # type: ignore[arg-type] # noqa: DTZ001

    def test_aware_source_updated_is_normalised_to_utc(self) -> None:
        book = RawBook(
            **raw_kwargs(source_updated=datetime(2026, 8, 11, 10, 0, tzinfo=UTC))  # type: ignore[arg-type]
        )

        assert book.source_updated is not None
        assert book.source_updated.tzinfo is UTC

    def test_is_immutable(self) -> None:
        book = RawBook(**raw_kwargs())  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            book.title = "Something else"  # type: ignore[misc]

    def test_raw_payload_is_required(self) -> None:
        # Provenance is not optional; book_sources.raw_payload is NOT NULL.
        kwargs = raw_kwargs()
        del kwargs["raw_payload"]

        with pytest.raises(ValidationError, match="raw_payload"):
            RawBook(**kwargs)  # type: ignore[arg-type]


def clean_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "source": SourceName.OPENLIBRARY,
        "source_id": "OL1234W",
        "identity_key": "isbn:9780553380163",
        "title": "A Brief History of Time",
        "normalised_title": "a brief history of time",
        "isbn13": "9780553380163",
        "raw_payload": {"key": "/works/OL1234W"},
    }
    return base | overrides


class TestCleanBook:
    def test_minimal_record_is_valid(self) -> None:
        book = CleanBook(**clean_kwargs())  # type: ignore[arg-type]

        assert book.identity_key == "isbn:9780553380163"
        assert book.published_year is None

    @pytest.mark.parametrize("bad", ["978055338016", "97805533801634", "978-055338016X"])
    def test_isbn13_must_be_thirteen_digits(self, bad: str) -> None:
        # Mirrors the CHECK constraint on books.isbn13.
        with pytest.raises(ValidationError, match="isbn13"):
            CleanBook(**clean_kwargs(isbn13=bad))  # type: ignore[arg-type]

    def test_isbn13_may_be_absent(self) -> None:
        # Gutendex supplies no ISBNs at all; this is the common case.
        book = CleanBook(**clean_kwargs(isbn13=None, identity_key="fallback:" + "a" * 64))  # type: ignore[arg-type]

        assert book.isbn13 is None

    @pytest.mark.parametrize("year", [1399, 2101])
    def test_published_year_outside_schema_range_is_rejected(self, year: int) -> None:
        with pytest.raises(ValidationError, match="published_year"):
            CleanBook(**clean_kwargs(published_year=year))  # type: ignore[arg-type]

    @pytest.mark.parametrize("year", [1400, 1988, 2100])
    def test_published_year_within_range_is_accepted(self, year: int) -> None:
        assert CleanBook(**clean_kwargs(published_year=year)).published_year == year  # type: ignore[arg-type]

    @pytest.mark.parametrize("count", [0, -1])
    def test_non_positive_page_count_is_rejected(self, count: int) -> None:
        with pytest.raises(ValidationError, match="page_count"):
            CleanBook(**clean_kwargs(page_count=count))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["en", "ENG", "english", "e1g"])
    def test_language_must_be_three_lowercase_letters(self, bad: str) -> None:
        # Mirrors the CHECK constraint on books.language (ISO 639-3).
        with pytest.raises(ValidationError, match="language"):
            CleanBook(**clean_kwargs(language=bad))  # type: ignore[arg-type]

    def test_language_eng_is_accepted(self) -> None:
        assert CleanBook(**clean_kwargs(language="eng")).language == "eng"  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "key",
        ["", "isbn:", "9780553380163", "unknown:abc", "fallback:tooshort"],
    )
    def test_identity_key_must_match_a_known_scheme(self, key: str) -> None:
        with pytest.raises(ValidationError, match="identity_key"):
            CleanBook(**clean_kwargs(identity_key=key))  # type: ignore[arg-type]

    def test_isbn_identity_key_must_agree_with_isbn13(self) -> None:
        # An identity key that disagrees with its own ISBN would split or
        # merge the wrong canonical books.
        with pytest.raises(ValidationError, match="identity_key"):
            CleanBook(**clean_kwargs(identity_key="isbn:9780306406157"))  # type: ignore[arg-type]

    def test_fallback_identity_key_requires_absent_isbn13(self) -> None:
        with pytest.raises(ValidationError, match="identity_key"):
            CleanBook(**clean_kwargs(identity_key="fallback:" + "b" * 64))  # type: ignore[arg-type]

    def test_is_immutable(self) -> None:
        book = CleanBook(**clean_kwargs())  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            book.title = "Other"  # type: ignore[misc]


class TestGutendexNativeFields:
    """Fields that only Gutendex supplies, and that the catalogue depends on.

    Gutendex carries no publication year, publisher, ISBN or page count, so it
    cannot feed the year-based analytics the catalogue was originally shaped
    around. What it does carry densely is author lifespan and download counts,
    which is why these are first-class fields rather than payload trivia.
    """

    def test_author_lifespan_is_captured(self) -> None:
        author = RawAuthor(name="Austen, Jane", birth_year=1775, death_year=1817)

        assert author.birth_year == 1775
        assert author.death_year == 1817

    def test_author_lifespan_is_optional(self) -> None:
        # Open Library and Google Books supply neither.
        author = RawAuthor(name="Anonymous")

        assert author.birth_year is None
        assert author.death_year is None

    @pytest.mark.parametrize(("birth", "death"), [(-750, -650), (-428, -348)])
    def test_bce_author_years_are_accepted(self, birth: int, death: int) -> None:
        # Homer is -750/-650 in Gutendex. A `ge=0` or `ge=1400` bound would
        # reject a real record from the primary source.
        author = RawAuthor(name="Homer", birth_year=birth, death_year=death)

        assert author.birth_year == birth
        assert author.death_year == death

    def test_death_before_birth_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="death_year"):
            RawAuthor(name="Impossible", birth_year=1900, death_year=1850)

    def test_equal_birth_and_death_year_is_accepted(self) -> None:
        # Died in infancy, or the source only knows one year for both.
        author = RawAuthor(name="Brief", birth_year=1900, death_year=1900)

        assert author.death_year == 1900

    @pytest.mark.parametrize("year", [-5000, 3000])
    def test_implausible_author_years_are_rejected(self, year: int) -> None:
        with pytest.raises(ValidationError, match="birth_year"):
            RawAuthor(name="Implausible", birth_year=year)

    def test_download_count_is_captured(self) -> None:
        book = RawBook(**raw_kwargs(download_count=183505))  # type: ignore[arg-type]

        assert book.download_count == 183505

    def test_download_count_is_optional(self) -> None:
        # Only Gutendex publishes one.
        assert RawBook(**raw_kwargs()).download_count is None  # type: ignore[arg-type]

    def test_negative_download_count_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="download_count"):
            RawBook(**raw_kwargs(download_count=-1))  # type: ignore[arg-type]

    def test_clean_book_carries_download_count_and_authors(self) -> None:
        book = CleanBook(
            **clean_kwargs(  # type: ignore[arg-type]
                download_count=46103,
                authors=[RawAuthor(name="Homer", birth_year=-750, death_year=-650)],
            )
        )

        assert book.download_count == 46103
        assert book.authors[0].birth_year == -750


class TestLanguagesArePreserved:
    """A source's language list survives extraction intact.

    Open Library returns one entry per edition of a work. Collapsing that to
    its first element tagged *The Picture of Dorian Gray* as Czech in a live
    run — the model has to keep what the source actually said and let transform
    decide, because only transform knows that an ambiguous list is unusable.
    """

    def test_multiple_languages_survive(self) -> None:
        book = RawBook(**raw_kwargs(languages=["eng", "cze", "fre"]))  # type: ignore[arg-type]

        assert book.languages == ["eng", "cze", "fre"]

    def test_a_single_language_is_still_a_list(self) -> None:
        book = RawBook(**raw_kwargs(languages=["en"]))  # type: ignore[arg-type]

        assert book.languages == ["en"]

    def test_absent_languages_default_to_empty(self) -> None:
        assert RawBook(**raw_kwargs()).languages == []  # type: ignore[arg-type]
