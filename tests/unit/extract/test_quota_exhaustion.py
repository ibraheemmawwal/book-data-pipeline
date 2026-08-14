"""Telling a spent allowance apart from momentary throttling.

Both arrive as HTTP 429 and they call for opposite responses. Per-minute
throttling clears in seconds and is worth waiting out. A daily allowance does
not clear until midnight, so every retry spends another request from the very
allowance that has run out.

This is not hypothetical arithmetic: Google's Books API caps a project at 1,000
requests a day and the limit cannot be raised — a request to increase it is
rejected outright with "Unsupported service". Retrying five times against that
turns a 1,000-book budget into a 200-book one.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pipeline.extract.base import (
    QuotaExhaustedError,
    SourceUnavailableError,
    quota_is_exhausted,
    request_with_retries,
)

URL = "https://example.test/volumes"


async def _no_wait(_seconds: float) -> None:
    """Collapse the backoff so these run in milliseconds."""
    return


def daily_limit_body() -> dict[str, object]:
    """The shape Google actually returns when the day's allowance is gone."""
    return {
        "error": {
            "code": 429,
            "message": "Quota exceeded",
            "errors": [
                {
                    "domain": "usageLimits",
                    "reason": "dailyLimitExceeded",
                    "message": "Daily Limit Exceeded",
                }
            ],
        }
    }


def rate_limit_body() -> dict[str, object]:
    """Momentary throttling: the same status, the opposite meaning."""
    return {
        "error": {
            "code": 429,
            "message": "Rate Limit Exceeded",
            "errors": [{"domain": "usageLimits", "reason": "rateLimitExceeded"}],
        }
    }


class TestRecognisingASpentAllowance:
    def test_a_daily_limit_is_recognised(self) -> None:
        response = httpx.Response(429, json=daily_limit_body())

        assert quota_is_exhausted(response) is True

    def test_momentary_throttling_is_not(self) -> None:
        response = httpx.Response(429, json=rate_limit_body())

        assert quota_is_exhausted(response) is False

    @pytest.mark.parametrize(
        "payload",
        [
            "not json at all",
            "[]",
            '{"error": "a string, not an object"}',
            "{}",
        ],
    )
    def test_an_unreadable_body_is_treated_as_throttling(self, payload: str) -> None:
        # The safe direction: retrying a spent allowance wastes requests, but
        # refusing to retry genuine throttling loses records we could have had.
        response = httpx.Response(429, content=payload)

        assert quota_is_exhausted(response) is False


class TestRetryBehaviour:
    async def _call(self, attempts: int = 5) -> httpx.Response:
        async def no_wait(_seconds: float) -> None:
            return None

        async with httpx.AsyncClient() as client:
            return await request_with_retries(
                client, "GET", URL, max_attempts=attempts, sleep=no_wait, source="googlebooks"
            )

    @respx.mock
    async def test_a_spent_allowance_is_not_retried(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(429, json=daily_limit_body()))

        with pytest.raises(QuotaExhaustedError):
            await self._call()

        # One request, not five. The other four could not have succeeded and
        # each would have cost another slice of the same exhausted budget.
        assert route.call_count == 1

    @respx.mock
    async def test_momentary_throttling_is_still_retried(self) -> None:
        # The behaviour that must not regress: not retrying a 429 is how a
        # polite client gets itself blocked.
        route = respx.get(URL).mock(return_value=httpx.Response(429, json=rate_limit_body()))

        with pytest.raises(SourceUnavailableError):
            await self._call(attempts=3)

        assert route.call_count == 3

    @respx.mock
    async def test_throttling_that_clears_still_succeeds(self) -> None:
        respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, json=rate_limit_body()),
                httpx.Response(200, json={"items": []}),
            ]
        )

        assert (await self._call()).status_code == 200

    @respx.mock
    async def test_the_error_says_which_kind_it_was(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(429, json=daily_limit_body()))

        with pytest.raises(QuotaExhaustedError, match="allowance exhausted"):
            await self._call()

    def test_it_remains_a_source_failure(self) -> None:
        # Callers that only know about SourceUnavailableError must keep working.
        assert issubclass(QuotaExhaustedError, SourceUnavailableError)


def modern_quota_body() -> dict[str, object]:
    """Google's newer envelope: no legacy reason code, only a status."""
    return {
        "error": {
            "code": 429,
            "message": "Quota exceeded for quota metric 'Queries'",
            "status": "RESOURCE_EXHAUSTED",
        }
    }


class TestTheEnvelopeGoogleActuallySends:
    def test_the_modern_status_is_recognised(self) -> None:
        # The legacy reason set matched neither this nor rateLimitExceeded, so
        # every 429 fell through to the retry path.
        assert quota_is_exhausted(httpx.Response(429, json=modern_quota_body()))

    def test_momentary_throttling_still_is_not(self) -> None:
        # Short-circuiting on the first of these would let one blip disable the
        # source for a whole run.
        assert not quota_is_exhausted(httpx.Response(429, json=rate_limit_body()))

    @respx.mock
    async def test_it_stops_on_the_first_response(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(429, json=modern_quota_body()))

        with pytest.raises(QuotaExhaustedError):
            await request_with_retries(
                httpx.AsyncClient(), "GET", URL, max_attempts=5, sleep=_no_wait
            )

        assert route.call_count == 1


class TestSurvivingEveryRetryIsItselfTheAnswer:
    """The failure that cost ~33,000 requests against a 1,000/day cap.

    6,662 candidates each paid five requests to be told the allowance was
    spent, because the body never said so in a form the reason set matched and
    the generic error left the run's budget intact. Five 429s in a row is a
    spent allowance by any reading worth acting on.
    """

    @respx.mock
    async def test_an_unrecognised_429_becomes_exhaustion(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(429, json=rate_limit_body()))

        # Not SourceUnavailableError: the caller has to be able to tell that
        # asking again this run is pointless, and only this type says so.
        with pytest.raises(QuotaExhaustedError):
            await request_with_retries(
                httpx.AsyncClient(), "GET", URL, max_attempts=5, sleep=_no_wait
            )

    @respx.mock
    async def test_the_retries_still_happen_first(self) -> None:
        # A 429 that clears must still be waited out; the point is what happens
        # when it does not.
        route = respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, json=rate_limit_body()),
                httpx.Response(429, json=rate_limit_body()),
                httpx.Response(200, json={}),
            ]
        )

        response = await request_with_retries(
            httpx.AsyncClient(), "GET", URL, max_attempts=5, sleep=_no_wait
        )

        assert response.status_code == 200
        assert route.call_count == 3

    @respx.mock
    async def test_a_non_429_exhaustion_stays_a_plain_failure(self) -> None:
        # A run of 503s is the source failing, not an allowance being spent,
        # and must not retire the budget for the rest of the run.
        respx.get(URL).mock(return_value=httpx.Response(503))

        with pytest.raises(SourceUnavailableError) as caught:
            await request_with_retries(
                httpx.AsyncClient(), "GET", URL, max_attempts=3, sleep=_no_wait
            )

        assert not isinstance(caught.value, QuotaExhaustedError)
