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
from pydantic import SecretStr

from pipeline.config import Settings
from pipeline.extract.goodreads import GoodreadsExtractor
from pipeline.extract.resolver import Attempt, Budget, CatalogueResolver, Outcome
from pipeline.models.domain import CandidateBook, SourceName

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
AUTOCOMPLETE = "https://www.goodreads.com/book/auto_complete"
OL_SEARCH = "https://openlibrary.org/search.json"
GB_VOLUMES = "https://www.googleapis.com/books/v1/volumes"
GUTENDEX = "https://gutendex.com/books"


def mock_live_fallbacks_empty() -> None:
    """Make every documented fallback answer with nothing.

    Explicit rather than implicit: an unmocked call raises, which is how these
    tests notice when a change starts reaching for a source it should not.
    """
    respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
    respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
    respx.get(GUTENDEX).mock(return_value=httpx.Response(200, json={"next": None, "results": []}))


def autocomplete_payload() -> Any:
    with (FIXTURES / "goodreads_autocomplete.json").open() as handle:
        return json.load(handle)


@pytest.fixture
def accepted(settings: Settings) -> Settings:
    """Both gates open *and* Goodreads on the ingestion path.

    The last flag is separate on purpose: opening the gates permits targeted
    use, and putting the source on the path every candidate takes is a further
    decision. These tests exercise that path, so they opt into it explicitly.
    """
    return settings.model_copy(
        update={
            "goodreads_enabled": True,
            "goodreads_unofficial_source_accepted": True,
            "goodreads_in_resolution": True,
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
        mock_live_fallbacks_empty()

        result = await CatalogueResolver(accepted, goodreads=goodreads).resolve(candidate())

        assert outcome_for(result.attempts, SourceName.GOODREADS) is Outcome.NO_MATCH
        assert not result.resolved

    @respx.mock
    async def test_a_block_is_recorded_as_unavailable(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(403))
        mock_live_fallbacks_empty()

        result = await CatalogueResolver(accepted, goodreads=goodreads).resolve(candidate())

        assert outcome_for(result.attempts, SourceName.GOODREADS) is Outcome.UNAVAILABLE

    @respx.mock
    async def test_an_open_circuit_skips_every_later_candidate(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        # One upstream outage must not become thousands of failing calls.
        route = respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(403))
        mock_live_fallbacks_empty()
        resolver = CatalogueResolver(accepted, goodreads=goodreads)

        await resolver.resolve(candidate(candidate_key="a"))
        second = await resolver.resolve(candidate(candidate_key="b"))

        assert route.call_count == 1
        attempt = next(a for a in second.attempts if a.source is SourceName.GOODREADS)
        assert attempt.outcome is Outcome.SKIPPED
        assert attempt.fallback_reason == "circuit open for this run"


class TestGates:
    """Why Goodreads was not used, when it is on the ingestion path.

    Recorded rather than silent: a source that was configured off and a source
    that had nothing to say must be distinguishable afterwards.
    """

    @pytest.fixture
    def on_path(self, settings: Settings) -> Settings:
        return settings.model_copy(update={"goodreads_in_resolution": True})

    async def test_an_unaccepted_source_is_skipped_with_a_reason(self, on_path: Settings) -> None:
        # Not an error: a clean clone runs the documented path without it.
        mock_live_fallbacks_empty()
        resolver = CatalogueResolver(on_path, goodreads=GoodreadsExtractor(on_path))

        result = await resolver.resolve(candidate())

        attempt = next(a for a in result.attempts if a.source is SourceName.GOODREADS)
        assert attempt.outcome is Outcome.SKIPPED
        assert "disabled" in (attempt.fallback_reason or "")

    async def test_no_adapter_configured_is_skipped(self, on_path: Settings) -> None:
        mock_live_fallbacks_empty()
        result = await CatalogueResolver(on_path).resolve(candidate())

        assert outcome_for(result.attempts, SourceName.GOODREADS) is Outcome.SKIPPED


class TestRetainedDiscovery:
    async def test_the_discovery_payload_becomes_an_observation(self, settings: Settings) -> None:
        # Free: it cost a dump read, not a request.
        mock_live_fallbacks_empty()
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
        mock_live_fallbacks_empty()
        result = await CatalogueResolver(settings).resolve(
            candidate(discovery_payload={"no_key": True})
        )

        assert outcome_for(result.attempts, SourceName.OPENLIBRARY) is Outcome.CONTRACT_FAILURE
        assert result.rejections

    async def test_no_payload_is_recorded_as_skipped(self, settings: Settings) -> None:
        mock_live_fallbacks_empty()
        result = await CatalogueResolver(settings).resolve(candidate())

        attempt = next(a for a in result.attempts if a.source is SourceName.OPENLIBRARY)
        assert attempt.outcome is Outcome.SKIPPED


class TestObservability:
    async def test_every_documented_source_attempt_is_recorded(self, settings: Settings) -> None:
        """A source that was skipped and one that had nothing must differ.

        Goodreads is absent because ingestion no longer consults it; when it
        does — goodreads_in_resolution — its attempts are recorded the same way,
        which TestGates covers.
        """
        mock_live_fallbacks_empty()
        result = await CatalogueResolver(settings).resolve(candidate())

        assert {a.source for a in result.attempts} >= {
            SourceName.OPENLIBRARY,
            SourceName.GOOGLEBOOKS,
        }

    async def test_attempts_carry_the_candidate_key(self, settings: Settings) -> None:
        mock_live_fallbacks_empty()
        result = await CatalogueResolver(settings).resolve(candidate(candidate_key="/works/OL9W"))

        assert all(a.candidate_key == "/works/OL9W" for a in result.attempts)

    async def test_duration_is_measured_from_an_injectable_clock(self, settings: Settings) -> None:
        mock_live_fallbacks_empty()
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


class TestDocumentedFallbacks:
    """Open Library, Google Books and Gutendex, under budget."""

    @respx.mock
    async def test_google_books_answers_when_nothing_else_did(self, settings: Settings) -> None:
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GB_VOLUMES).mock(
            return_value=httpx.Response(
                200,
                json={"items": [{"id": "gb1", "volumeInfo": {"title": "A Game of Thrones"}}]},
            )
        )
        configured = settings.model_copy(update={"googlebooks_api_key": SecretStr("k")})

        result = await CatalogueResolver(configured).resolve(candidate())

        assert result.observations[0].source is SourceName.GOOGLEBOOKS
        assert outcome_for(result.attempts, SourceName.GOOGLEBOOKS) is Outcome.RESOLVED

    @respx.mock
    async def test_gutendex_is_only_reached_when_all_else_fails(self, settings: Settings) -> None:
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
        gutendex = respx.get(GUTENDEX).mock(
            return_value=httpx.Response(
                200,
                json={"next": None, "results": [{"id": 1, "title": "A Game of Thrones"}]},
            )
        )
        configured = settings.model_copy(update={"googlebooks_api_key": SecretStr("k")})

        result = await CatalogueResolver(configured).resolve(candidate())

        assert gutendex.called
        assert result.observations[0].source is SourceName.GUTENDEX

    @respx.mock
    async def test_gutendex_is_skipped_when_something_already_resolved(
        self, settings: Settings
    ) -> None:
        # A last resort that runs anyway is not a last resort.
        gutendex = respx.get(GUTENDEX).mock(return_value=httpx.Response(200, json={}))

        await CatalogueResolver(settings).resolve(
            candidate(discovery_payload={"key": "/works/OL1W", "title": "T"})
        )

        assert not gutendex.called

    @respx.mock
    async def test_live_fallbacks_are_skipped_when_goodreads_resolved(
        self, accepted: Settings, goodreads: GoodreadsExtractor
    ) -> None:
        # Filling gaps is not the same as re-answering an answered question;
        # spending budget here would starve the candidates that need it.
        respx.get(AUTOCOMPLETE).mock(return_value=httpx.Response(200, json=autocomplete_payload()))
        respx.get(url__regex=r".*/book/show/.*").mock(return_value=httpx.Response(404))
        respx.get(url__regex=r".*/work/editions/.*").mock(return_value=httpx.Response(404))
        gb = respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={}))

        result = await CatalogueResolver(accepted, goodreads=goodreads).resolve(candidate())

        assert result.resolved
        assert not gb.called


class TestBudgetEnforcement:
    @respx.mock
    async def test_an_exhausted_budget_records_a_skip(self, settings: Settings) -> None:
        # Budget exhaustion is observable, not permission to keep going.
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        capped = settings.model_copy(
            update={
                "openlibrary_max_fallback_queries_per_run": 1,
                "googlebooks_api_key": SecretStr("k"),
            }
        )
        resolver = CatalogueResolver(capped)

        await resolver.resolve(candidate(candidate_key="a"))
        second = await resolver.resolve(candidate(candidate_key="b"))

        attempt = next(
            a for a in second.attempts if a.source is SourceName.OPENLIBRARY and a.attempt_no == 2
        )
        assert attempt.outcome is Outcome.SKIPPED
        assert "budget" in (attempt.fallback_reason or "")

    @respx.mock
    async def test_the_budget_stops_further_requests(self, settings: Settings) -> None:
        route = respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        capped = settings.model_copy(update={"openlibrary_max_fallback_queries_per_run": 1})
        resolver = CatalogueResolver(capped)

        for key in ("a", "b", "c"):
            await resolver.resolve(candidate(candidate_key=key))

        assert route.call_count == 1

    @respx.mock
    async def test_a_missing_api_key_is_a_skip_not_an_outage(self, settings: Settings) -> None:
        # Our configuration gap, not the provider's. Recording it as
        # unavailable would send someone hunting a fault at Google.
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )

        keyless = settings.model_copy(update={"googlebooks_api_key": None})

        result = await CatalogueResolver(keyless).resolve(candidate())

        attempt = next(a for a in result.attempts if a.source is SourceName.GOOGLEBOOKS)
        assert attempt.outcome is Outcome.SKIPPED
        assert "api key" in (attempt.fallback_reason or "").lower()

    @respx.mock
    async def test_an_unavailable_fallback_is_recorded_and_not_fatal(
        self, settings: Settings
    ) -> None:
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(503))
        respx.get(GB_VOLUMES).mock(return_value=httpx.Response(200, json={"totalItems": 0}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        configured = settings.model_copy(
            update={"googlebooks_api_key": SecretStr("k"), "http_max_attempts": 1}
        )

        result = await CatalogueResolver(configured).resolve(candidate())

        # attempt_no 2 is the live lookup; 1 is the retained-discovery leg.
        live = next(
            a for a in result.attempts if a.source is SourceName.OPENLIBRARY and a.attempt_no == 2
        )
        assert live.outcome is Outcome.UNAVAILABLE
        assert not result.resolved


class TestMissingCredentialHandling:
    @respx.mock
    async def test_a_credential_error_mid_fetch_is_a_skip(self, settings: Settings) -> None:
        # GoogleBooksExtractor raises before any request when the key is
        # absent. Recording that as unavailable would send someone hunting a
        # fault at Google rather than at our configuration.
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        # Enabled, budgeted, but keyless: skip_reason lets it through and the
        # extractor raises MissingCredentialError from inside fetch.
        keyless = settings.model_copy(update={"googlebooks_api_key": None})
        resolver = CatalogueResolver(keyless)
        object.__setattr__(
            resolver, "_settings", keyless.model_copy(update={"googlebooks_enabled": True})
        )

        result = await resolver.resolve(candidate())

        googlebooks = next(a for a in result.attempts if a.source is SourceName.GOOGLEBOOKS)
        assert googlebooks.outcome is Outcome.SKIPPED


class TestEnrichmentMode:
    """Querying documented sources for a book something already resolved.

    Gap-filling is the right default and it has a consequence worth naming: a
    book resolved by exactly one source has no cross-source provenance and can
    never disagree with anything. A catalogue built purely that way cannot
    answer "do the sources agree", which is the question this pipeline exists
    to make answerable.
    """

    @pytest.fixture(autouse=True)
    def _mocked(self) -> None:
        # Enrichment reaches these on purpose, so they must answer.
        mock_live_fallbacks_empty()

    async def test_by_default_a_resolved_candidate_skips_the_documented_apis(
        self, settings: Settings
    ) -> None:
        resolver = CatalogueResolver(settings)
        candidate = CandidateBook(
            candidate_key="/works/OL1W",
            title="Dune",
            discovery_payload={"key": "/works/OL1W", "title": "Dune"},
        )

        result = await resolver.resolve(candidate)

        assert result.resolved
        googlebooks = [a for a in result.attempts if a.source is SourceName.GOOGLEBOOKS]
        assert googlebooks == []

    async def test_enrichment_queries_them_anyway(self, settings: Settings) -> None:
        resolver = CatalogueResolver(
            settings.model_copy(update={"enrich_from_documented_sources": True})
        )
        candidate = CandidateBook(
            candidate_key="/works/OL1W",
            title="Dune",
            discovery_payload={"key": "/works/OL1W", "title": "Dune"},
        )

        result = await resolver.resolve(candidate)

        assert result.resolved
        # Attempted, whatever the outcome — the point is that it was tried.
        assert any(a.source is SourceName.GOOGLEBOOKS for a in result.attempts)

    async def test_enrichment_still_respects_the_budget(self, settings: Settings) -> None:
        # The budget is what stops enrichment turning one run into a quota
        # incident; without it this mode would query every source for every
        # candidate indefinitely.
        settings = settings.model_copy(
            update={
                "enrich_from_documented_sources": True,
                "googlebooks_max_fallback_queries_per_run": 1,
            }
        )
        resolver = CatalogueResolver(settings)

        outcomes = []
        for index in range(3):
            candidate = CandidateBook(
                candidate_key=f"/works/OL{index}W",
                title=f"Book {index}",
                discovery_payload={"key": f"/works/OL{index}W", "title": f"Book {index}"},
            )
            result = await resolver.resolve(candidate)
            outcomes += [a.outcome for a in result.attempts if a.source is SourceName.GOOGLEBOOKS]

        assert outcomes.count(Outcome.SKIPPED) >= 2


class TestASpentDailyAllowance:
    """What happens to the rest of the run once a source's day is gone.

    Google's Books API allows a project 1,000 requests a day and will not raise
    it. Continuing to ask costs one request per candidate to be told the same
    thing, and those requests come out of tomorrow's allowance if the run
    crosses midnight — so the source retires for the run instead.
    """

    @staticmethod
    def _daily_limit() -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "errors": [{"domain": "usageLimits", "reason": "dailyLimitExceeded"}],
                }
            },
        )

    @respx.mock
    async def test_the_source_is_not_asked_again(self, settings: Settings) -> None:
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        google = respx.get(GB_VOLUMES).mock(return_value=self._daily_limit())
        resolver = CatalogueResolver(
            settings.model_copy(update={"googlebooks_api_key": SecretStr("k")})
        )

        for key in ("a", "b", "c", "d"):
            await resolver.resolve(candidate(candidate_key=key))

        # Once, for the candidate that discovered it. Without this the four
        # candidates cost four requests, and a 500-book run costs 500.
        assert google.call_count == 1

    @respx.mock
    async def test_later_candidates_record_why(self, settings: Settings) -> None:
        respx.get(OL_SEARCH).mock(return_value=httpx.Response(200, json={"docs": []}))
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        respx.get(GB_VOLUMES).mock(return_value=self._daily_limit())
        resolver = CatalogueResolver(
            settings.model_copy(update={"googlebooks_api_key": SecretStr("k")})
        )

        await resolver.resolve(candidate(candidate_key="first"))
        later = await resolver.resolve(candidate(candidate_key="second"))

        attempt = next(a for a in later.attempts if a.source is SourceName.GOOGLEBOOKS)
        # A skip with a reason, not silence: a run where Google Books stopped
        # contributing should say so rather than just thin out.
        assert attempt.outcome is Outcome.SKIPPED

    @respx.mock
    async def test_the_other_sources_carry_on(self, settings: Settings) -> None:
        # One source's allowance ending is not the run ending.
        respx.get(GB_VOLUMES).mock(return_value=self._daily_limit())
        respx.get(GUTENDEX).mock(
            return_value=httpx.Response(200, json={"next": None, "results": []})
        )
        openlibrary = respx.get(OL_SEARCH).mock(
            return_value=httpx.Response(
                200, json={"docs": [{"key": "/works/OL1W", "title": "Dune"}]}
            )
        )
        resolver = CatalogueResolver(
            settings.model_copy(update={"googlebooks_api_key": SecretStr("k")})
        )

        for key in ("a", "b"):
            result = await resolver.resolve(candidate(candidate_key=key))

        assert openlibrary.call_count == 2
        assert result.resolved
