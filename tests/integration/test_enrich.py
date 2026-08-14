"""Completing Goodreads records that arrived without their detail.

An export supplies a title, its authors, a rating and a cover; the year, the
ISBN and the page count live on the book's own page. These tests pin what a
pass over that backlog does, against the real schema — including the part that
matters most, which is that completing a record must not create a second one.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Connection, Engine, select

from pipeline.config import Settings
from pipeline.enrich import count_unenriched, enrich_goodreads, find_unenriched
from pipeline.extract.base import Rejected
from pipeline.extract.goodreads import GoodreadsNotAcceptedError
from pipeline.load import CatalogueLoader
from pipeline.models.db import books
from pipeline.models.domain import CleanBook, RawBook, SourceName
from pipeline.transform import canonicalise

pytestmark = pytest.mark.integration


def _clean(record: RawBook) -> CleanBook:
    result = canonicalise(record)
    assert isinstance(result, CleanBook), result
    return result


def imported(source_id: str, *, title: str = "Imported Book") -> CleanBook:
    """A record as the export leaves it: no year, no ISBN, no detail block."""
    return _clean(
        RawBook(
            source=SourceName.GOODREADS,
            source_id=source_id,
            title=title,
            raw_payload={
                "bookId": source_id,
                "title": title,
                "bookTitleBare": title,
                "author": {"name": "Some Author"},
                "_export": {"source_file": "standalone.json"},
            },
        )
    )


def settings_for(url: str, *, accepted: bool = True) -> Settings:
    return Settings(  # type: ignore[call-arg]
        database_url=url,
        openlibrary_contact_email="t@example.com",
        goodreads_enabled=True,
        goodreads_unofficial_source_accepted=accepted,
    )


def _answers(observation_updates: dict[str, Any] | None) -> Any:
    """A stubbed detail fetch. None means the page gave nothing.

    The blocks matter as much as the model fields: canonical values are
    recomputed by replaying the stored payload, so a year set only on the
    object survives until the next recompute and then vanishes. The real
    enrichment stores _detail and _edition, and the replay reads the year out
    of _edition — so the stub does the same.
    """

    async def fake(_self: Any, _client: Any, observation: RawBook) -> RawBook | None:
        if observation_updates is None:
            return None
        payload = {
            **observation.raw_payload,
            "_detail": {"json_ld": {}},
            "_edition": {
                "published": observation_updates.get("published"),
                "isbn13": (observation_updates.get("isbns") or [None])[0],
            },
        }
        return observation.model_copy(update={**observation_updates, "raw_payload": payload})

    return fake


class TestFindingTheBacklog:
    def test_imported_records_are_pending(self, engine: Engine) -> None:
        CatalogueLoader().load(engine, [imported("gr-1"), imported("gr-2")])

        assert count_unenriched(engine) == 2
        assert {r["source_id"] for r in find_unenriched(engine, limit=10)} == {"gr-1", "gr-2"}

    def test_a_record_with_detail_is_not_pending(self, engine: Engine) -> None:
        record = imported("gr-3")
        record.raw_payload["_detail"] = {"json_ld": {}}
        CatalogueLoader().load(engine, [record])

        assert count_unenriched(engine) == 0

    def test_the_limit_is_honoured(self, engine: Engine) -> None:
        CatalogueLoader().load(engine, [imported(f"gr-{n}") for n in range(5)])

        assert len(find_unenriched(engine, limit=2)) == 2

    def test_the_oldest_are_taken_first(self, engine: Engine) -> None:
        """Otherwise a run revisits the head of the queue forever.

        Newest-first would re-fetch whatever arrived most recently and never
        reach the records imported first, which is most of the backlog.
        """
        loader = CatalogueLoader()
        loader.load(engine, [imported("older")])
        loader.load(engine, [imported("newer")])

        assert find_unenriched(engine, limit=1)[0]["source_id"] == "older"


class TestEnriching:
    def test_it_completes_a_record_without_creating_a_second(
        self, engine: Engine, connection: Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure this whole flow has to avoid.

        The observation keeps its own (source, source_id), so the loader
        attaches it to the book it already belongs to. A re-keyed answer would
        give the catalogue a duplicate instead of a completed record — which is
        exactly what happened once in the contested flow.
        """
        CatalogueLoader().load(engine, [imported("gr-9")])
        before = connection.execute(select(books.c.id)).scalars().all()

        monkeypatch.setattr(
            "pipeline.extract.goodreads.GoodreadsExtractor.enrich_by_id",
            _answers({"published": "1979", "isbns": ["9780345391803"], "page_count": 224}),
        )
        report = enrich_goodreads(settings_for(str(engine.url)), limit=10, engine=engine)

        assert report.enriched == 1
        assert connection.execute(select(books.c.id)).scalars().all() == before

    def test_the_year_actually_lands(
        self, engine: Engine, connection: Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        CatalogueLoader().load(engine, [imported("gr-10")])
        monkeypatch.setattr(
            "pipeline.extract.goodreads.GoodreadsExtractor.enrich_by_id",
            _answers({"published": "1965"}),
        )

        enrich_goodreads(settings_for(str(engine.url)), limit=10, engine=engine)

        year = connection.execute(
            select(books.c.published_year).where(books.c.published_year.is_not(None))
        ).scalar_one()
        assert year == 1965

    def test_a_page_that_gives_nothing_is_counted_not_fatal(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        CatalogueLoader().load(engine, [imported("gr-11")])
        monkeypatch.setattr(
            "pipeline.extract.goodreads.GoodreadsExtractor.enrich_by_id", _answers(None)
        )

        report = enrich_goodreads(settings_for(str(engine.url)), limit=10, engine=engine)

        assert (report.queried, report.enriched, report.unchanged) == (1, 0, 1)

    def test_an_enriched_record_leaves_the_backlog(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Otherwise every run fetches the same records and the backlog never
        # shrinks, however many pages are fetched.
        CatalogueLoader().load(engine, [imported("gr-12")])
        monkeypatch.setattr(
            "pipeline.extract.goodreads.GoodreadsExtractor.enrich_by_id",
            _answers({"published": "1984"}),
        )

        enrich_goodreads(settings_for(str(engine.url)), limit=10, engine=engine)

        assert count_unenriched(engine) == 0

    def test_the_pending_count_is_the_whole_backlog_not_the_slice(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A report saying "2 pending" after a 2-record slice of 5 would suggest
        # the work was nearly done.
        CatalogueLoader().load(engine, [imported(f"gr-2{n}") for n in range(5)])
        monkeypatch.setattr(
            "pipeline.extract.goodreads.GoodreadsExtractor.enrich_by_id", _answers(None)
        )

        report = enrich_goodreads(settings_for(str(engine.url)), limit=2, engine=engine)

        assert report.pending == 5
        assert report.queried == 2


class TestGatesAndEmptiness:
    def test_it_refuses_when_the_source_is_not_accepted(self, engine: Engine) -> None:
        # A bulk backlog is exactly where the acknowledgement matters most.
        CatalogueLoader().load(engine, [imported("gr-13")])

        with pytest.raises(GoodreadsNotAcceptedError):
            enrich_goodreads(settings_for(str(engine.url), accepted=False), engine=engine)

    def test_an_empty_backlog_opens_no_run(self, engine: Engine) -> None:
        report = enrich_goodreads(settings_for(str(engine.url)), engine=engine)

        assert report.pending == 0
        assert report.run_id is None

    def test_a_failure_mid_pass_still_closes_the_run(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run row left open is indistinguishable from work in progress."""
        CatalogueLoader().load(engine, [imported("gr-14")])

        async def explode(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("source fell over")

        monkeypatch.setattr("pipeline.extract.goodreads.GoodreadsExtractor.enrich_by_id", explode)

        with pytest.raises(RuntimeError):
            enrich_goodreads(settings_for(str(engine.url)), engine=engine)


class TestWhenTheSourceStopsAnswering:
    """The three ways a record can fail to complete, kept apart.

    A source that refuses us, a payload that cannot be read, and an answer that
    will not canonicalise are different problems with different remedies, and a
    report that merges them tells an operator nothing.
    """

    def test_an_open_circuit_stops_the_pass(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Continuing to ask a source that has refused us is the thing the
        circuit exists to prevent, and a backlog run is where it would do the
        most damage."""
        CatalogueLoader().load(engine, [imported(f"gr-3{n}") for n in range(4)])

        monkeypatch.setattr(
            "pipeline.extract.goodreads.GoodreadsExtractor.circuit_open",
            property(lambda _self: True),
        )
        report = enrich_goodreads(settings_for(str(engine.url)), limit=10, engine=engine)

        assert report.queried == 0
        assert any("circuit" in error for error in report.errors)

    def test_an_unreadable_payload_is_counted_as_failed(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Not "unchanged": nothing was asked of the source, so the record is
        # ours to fix rather than theirs.
        CatalogueLoader().load(engine, [imported("gr-40")])
        monkeypatch.setattr(
            "pipeline.enrich.map_payload",
            lambda _source, _payload: Rejected(
                source=SourceName.GOODREADS,
                source_id="gr-40",
                raw_payload={},
                rejection_code="invalid_record",
                detail="unreadable",
            ),
        )

        report = enrich_goodreads(settings_for(str(engine.url)), limit=10, engine=engine)

        assert (report.failed, report.queried) == (1, 0)

    def test_an_answer_that_will_not_canonicalise_is_reported(
        self, engine: Engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        CatalogueLoader().load(engine, [imported("gr-41")])
        monkeypatch.setattr(
            "pipeline.extract.goodreads.GoodreadsExtractor.enrich_by_id",
            _answers({"published": "1999"}),
        )
        monkeypatch.setattr(
            "pipeline.enrich.canonicalise",
            lambda _observation: Rejected(
                source=SourceName.GOODREADS,
                source_id="gr-41",
                raw_payload={},
                rejection_code="invalid_record",
                detail="no usable title",
            ),
        )

        report = enrich_goodreads(settings_for(str(engine.url)), limit=10, engine=engine)

        assert report.failed == 1
        assert report.errors
        assert "gr-41" in report.errors[0]
