"""The catalogue resolver.

Fallback lives in the resolver rather than inside any adapter, so that which
source answered, how long it took, and why the others were not used are all
observable. With an unofficial primary source, that visibility is the point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pipeline.config import Settings
from pipeline.extract.goodreads import GoodreadsExtractor
from pipeline.extract.resolver import Attempt, Budget, CatalogueResolver, Outcome
from pipeline.models.domain import CandidateBook, SourceName

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
AUTOCOMPLETE = "https://www.goodreads.com/book/auto_complete"


def autocomplete_payload() -> Any:
    with (FIXTURES / "goodreads_autocomplete.json").open() as handle:
        return json.load(handle)


@pytest.fixture
def accepted(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "goodreads_enabled": True,
            "goodreads_unofficial_source_accepted": True,
            "goodreads_circuit_failure_threshold": 1,
        }
    )


@pytest.fixture
def goodreads(accepted: Settings) -> GoodreadsExtractor:
    async def no_wait(_: float) -> None:
        return None

    return GoodreadsExtractor(accepted, sleep=no_wait)


def candidate(**overrides: Any) -> CandidateBook:
    base: dict[str, Any] = {
        "candidate_key": "/works/OL1W",
        "title": "A Game of Thrones",
        "authors": ["George R.R. Martin"],
    }
    return CandidateBook(**(base | overrides))


def outcome_for(attempts: list[Attempt], source: SourceName) -> Outcome | None:
    return next((a.outcome for a in attempts if a.source is source), None)


class TestBudget:
    def test_spending_is_capped(self) -> None:
        budget = Budget(limit=2)

        assert budget.try_spend()
        assert budget.try_spend()
        assert not budget.try_spend()

    def test_exhaustion_is_visible(self) -> None:
        budget = Budget(limit=1)
        budget.try_spend()

        assert budget.exhausted

    def test_a_zero_budget_permits_nothing(self) -> None:
        # A source can be switched off by budget alone.
        assert not Budget(limit=0).try_spend()


class TestGoodreadsFirst:
    @respx.mock
    async def test_goodreads_resolves_the_candidate(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=autocomplete_payload()))
        respx.get(url__regex=r".*/book/show/.*").mock(return_value=httpx.Response(404))
        respx.get(url__regex=r".*/work/editions/.*").mock(return_value=httpx.Response(404))

        result = await CatalogueResolver(accepted, goodreads=goodreads).resolve(candidate())

        assert result.resolved
        assert result.observations[0].source is SourceName.GOODREADS
        assert outcome_for(result.attempts, SourceName.GOODREADS) is Outcome.RESOLVED

    @respx.mock
    async def test_an_empty_autocomplete_is_recorded_as_no_match(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        # The source answered and had nothing — distinct from never answering.
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=[]))

        result = await CatalogueResolver(accepted, goodreads=goodreads).resolve(candidate())

        assert outcome_for(result.attempts, SourceName.GOODREADS) is Outcome.NO_MATCH
        assert not result.resolved

    @respx.mock
    async def test_a_block_is_recorded_as_unavailable(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(403))

        result = await CatalogueResolver(accepted, goodreads=goodreads).resolve(candidate())

        assert outcome_for(result.attempts, SourceName.GOODREADS) is Outcome.UNAVAILABLE

    @respx.mock
    async def test_an_open_circuit_skips_every_later_candidate(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        # One upstream outage must not become thousands of failing calls.
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(403))
        resolver = CatalogueResolver(accepted, goodreads=goodreads)

        await resolver.resolve(candidate(candidate_key="a"))
        second = await resolver.resolve(candidate(candidate_key="b"))

        assert route.call_count == 1
        attempt = next(a for a in second.attempts if a.source is SourceName.GOODREADS)
        assert attempt.outcome is Outcome.SKIPPED
        assert attempt.fallback_reason == "circuit open for this run"


class TestGates:
    async def test_an_unaccepted_source_is_skipped_with_a_reason(self, settings: Settings) -> None:
        # Not an error: a clean clone runs the documented path without it.
        resolver = CatalogueResolver(settings, goodreads=GoodreadsExtractor(settings))

        result = await resolver.resolve(candidate())

        attempt = next(a for a in result.attempts if a.source is SourceName.GOODREADS)
        assert attempt.outcome is Outcome.SKIPPED
        assert "disabled" in (attempt.fallback_reason or "")

    async def test_no_adapter_configured_is_skipped(self, settings: Settings) -> None:
        result = await CatalogueResolver(settings).resolve(candidate())

        assert outcome_for(result.attempts, SourceName.GOODREADS) is Outcome.SKIPPED


class TestRetainedDiscovery:
    async def test_the_discovery_payload_becomes_an_observation(self, settings: Settings) -> None:
        # Free: it cost a dump read, not a request.
        result = await CatalogueResolver(settings).resolve(
            candidate(discovery_payload={"key": "/works/OL1W", "title": "A Game of Thrones"})
        )

        assert result.resolved
        assert result.observations[0].source is SourceName.OPENLIBRARY

    async def test_it_is_kept_even_when_goodreads_already_resolved(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        # "Goodreads first" is lookup order and field preference, not a rule
        # that other valid observations must be discarded.
        with respx.mock:
            respx.get(AUTOCOMPLETE).mock(
                return_value=httpx.Response(200, json=autocomplete_payload())
            )
            respx.get(url__regex=r".*/book/show/.*").mock(return_value=httpx.Response(404))
            respx.get(url__regex=r".*/work/editions/.*").mock(return_value=httpx.Response(404))
            result = await CatalogueResolver(accepted, goodreads=goodreads).resolve(
                candidate(discovery_payload={"key": "/works/OL1W", "title": "A Game of Thrones"})
            )

        assert {o.source for o in result.observations} == {
            SourceName.GOODREADS,
            SourceName.OPENLIBRARY,
        }

    async def test_a_payload_that_fails_validation_is_a_contract_failure(
        self, settings: Settings
    ) -> None:
        # Discovery proves a book exists, not that its fields are usable.
        result = await CatalogueResolver(settings).resolve(
            candidate(discovery_payload={"no_key": True})
        )

        assert outcome_for(result.attempts, SourceName.OPENLIBRARY) is Outcome.CONTRACT_FAILURE
        assert result.rejections

    async def test_no_payload_is_recorded_as_skipped(self, settings: Settings) -> None:
        result = await CatalogueResolver(settings).resolve(candidate())

        attempt = next(a for a in result.attempts if a.source is SourceName.OPENLIBRARY)
        assert attempt.outcome is Outcome.SKIPPED


class TestObservability:
    async def test_every_source_attempt_is_recorded(self, settings: Settings) -> None:
        result = await CatalogueResolver(settings).resolve(candidate())

        assert {a.source for a in result.attempts} >= {
            SourceName.GOODREADS,
            SourceName.OPENLIBRARY,
        }

    async def test_attempts_carry_the_candidate_key(self, settings: Settings) -> None:
        result = await CatalogueResolver(settings).resolve(candidate(candidate_key="/works/OL9W"))

        assert all(a.candidate_key == "/works/OL9W" for a in result.attempts)

    async def test_duration_is_measured_from_an_injectable_clock(self, settings: Settings) -> None:
        ticks = iter([0.0, 0.25, 1.0, 1.0, 1.0])
        resolver = CatalogueResolver(settings, clock=lambda: next(ticks))

        result = await resolver.resolve(
            candidate(discovery_payload={"key": "/works/OL1W", "title": "T"})
        )

        resolved = next(a for a in result.attempts if a.outcome is Outcome.RESOLVED)
        assert resolved.duration_ms == 250

    def test_budgets_are_exposed_for_run_reporting(self, settings: Settings) -> None:
        budgets = CatalogueResolver(settings).budgets

        assert set(budgets) == {
            SourceName.OPENLIBRARY,
            SourceName.GOOGLEBOOKS,
            SourceName.GUTENDEX,
        }
