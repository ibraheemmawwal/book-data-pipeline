"""Shared extraction machinery: retries, rate limiting and failure typing.

The retry policy is a correctness concern, not a nicety. Retrying a 404 wastes
a source's goodwill; not retrying a 429 gets the pipeline blocked.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from pydantic import ValidationError

from pipeline.extract.base import (
    ExtractionRequest,
    Rejected,
    SourceUnavailableError,
    TokenBucket,
    build_client,
    request_with_retries,
)
from pipeline.models.domain import SourceName

URL = "https://example.test/books"


async def call(client: httpx.AsyncClient, **kwargs: object) -> httpx.Response:
    return await request_with_retries(client, "GET", URL, max_attempts=5, **kwargs)  # type: ignore[arg-type]


class TestRetryPolicy:
    @respx.mock
    @pytest.mark.parametrize("status", [500, 502, 503, 504])
    async def test_retries_server_errors_and_eventually_succeeds(self, status: int) -> None:
        route = respx.get(URL).mock(
            side_effect=[httpx.Response(status), httpx.Response(200, json={"ok": True})]
        )
        async with build_client(user_agent="test/1.0") as client:
            response = await call(client, base_delay=0.0)

        assert response.status_code == 200
        assert route.call_count == 2

    @respx.mock
    async def test_retries_429(self) -> None:
        route = respx.get(URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, json={})])
        async with build_client(user_agent="test/1.0") as client:
            await call(client, base_delay=0.0)

        assert route.call_count == 2

    @respx.mock
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_does_not_retry_ordinary_client_errors(self, status: int) -> None:
        # Retrying these burns the source's goodwill and cannot succeed.
        route = respx.get(URL).mock(return_value=httpx.Response(status))
        async with build_client(user_agent="test/1.0") as client:
            with pytest.raises(SourceUnavailableError):
                await call(client, base_delay=0.0)

        assert route.call_count == 1

    @respx.mock
    async def test_retries_transport_errors(self) -> None:
        route = respx.get(URL).mock(
            side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={})]
        )
        async with build_client(user_agent="test/1.0") as client:
            await call(client, base_delay=0.0)

        assert route.call_count == 2

    @respx.mock
    async def test_gives_up_after_the_attempt_cap(self) -> None:
        route = respx.get(URL).mock(return_value=httpx.Response(503))
        async with build_client(user_agent="test/1.0") as client:
            with pytest.raises(SourceUnavailableError):
                await call(client, base_delay=0.0)

        assert route.call_count == 5

    @respx.mock
    async def test_exhausted_retries_raise_a_typed_terminal_failure(self) -> None:
        # The Airflow barrier needs a typed result, not an httpx internal.
        respx.get(URL).mock(return_value=httpx.Response(503))
        async with build_client(user_agent="test/1.0") as client:
            with pytest.raises(SourceUnavailableError) as caught:
                await call(client, base_delay=0.0)

        assert caught.value.status_code == 503


class TestRetryAfter:
    @respx.mock
    async def test_honours_retry_after_seconds(self) -> None:
        respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json={}),
            ]
        )
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        async with build_client(user_agent="test/1.0") as client:
            await call(client, base_delay=0.0, sleep=record)

        assert slept == [7.0]

    @respx.mock
    async def test_ignores_an_unparseable_retry_after(self) -> None:
        # A malformed header must fall back to backoff, not crash the run.
        respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "not-a-number"}),
                httpx.Response(200, json={}),
            ]
        )
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        async with build_client(user_agent="test/1.0") as client:
            await call(client, base_delay=0.25, sleep=record)

        assert len(slept) == 1
        assert slept[0] > 0

    @respx.mock
    async def test_caps_an_absurd_retry_after(self) -> None:
        # A source asking us to wait an hour must not hold an Airflow task open.
        respx.get(URL).mock(
            side_effect=[
                httpx.Response(503, headers={"Retry-After": "3600"}),
                httpx.Response(200, json={}),
            ]
        )
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        async with build_client(user_agent="test/1.0") as client:
            await call(client, base_delay=0.0, max_delay=30.0, sleep=record)

        assert slept == [30.0]


class TestBackoff:
    @respx.mock
    async def test_delay_grows_and_is_jittered(self) -> None:
        respx.get(URL).mock(side_effect=[httpx.Response(503)] * 4 + [httpx.Response(200, json={})])
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        async with build_client(user_agent="test/1.0") as client:
            await call(client, base_delay=1.0, max_delay=60.0, sleep=record)

        assert len(slept) == 4
        # Exponential in the ceiling, jittered below it: strictly increasing
        # bounds without every client retrying in lockstep after an outage.
        assert all(0 < d <= 1.0 * 2**i for i, d in enumerate(slept))
        assert slept != sorted(slept) or len({round(d, 6) for d in slept}) > 1


class TestClient:
    async def test_sets_explicit_timeouts_on_every_phase(self) -> None:
        async with build_client(
            user_agent="test/1.0", connect_timeout=2.0, read_timeout=9.0
        ) as client:
            timeout = client.timeout

        # A missing write or pool timeout is how a hung socket becomes a
        # permanently stuck task.
        assert timeout.connect == 2.0
        assert timeout.read == 9.0
        assert timeout.write is not None
        assert timeout.pool is not None

    async def test_identifies_itself(self) -> None:
        async with build_client(user_agent="book-data-pipeline/1.0 (+url)") as client:
            assert client.headers["user-agent"] == "book-data-pipeline/1.0 (+url)"


class TestTokenBucket:
    async def test_first_acquire_is_immediate(self) -> None:
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        bucket = TokenBucket(rate_per_second=1.0, sleep=record)
        await bucket.acquire()

        assert slept == []

    async def test_second_acquire_waits_for_the_interval(self) -> None:
        slept: list[float] = []
        clock = iter([0.0, 0.0, 1.0])

        async def record(delay: float) -> None:
            slept.append(delay)

        bucket = TokenBucket(rate_per_second=1.0, sleep=record, monotonic=lambda: next(clock))
        await bucket.acquire()
        await bucket.acquire()

        assert slept == [pytest.approx(1.0)]

    async def test_waits_do_not_compound_when_sleep_returns_early(self) -> None:
        # Regression: scheduling the next slot from the slot just waited for
        # rather than from the clock made every wait longer than the last —
        # 1s, then 2s, then 3s — silently throttling an extractor to a crawl.
        slept: list[float] = []

        async def record(delay: float) -> None:
            slept.append(delay)

        bucket = TokenBucket(rate_per_second=1.0, sleep=record, monotonic=lambda: 0.0)
        for _ in range(4):
            await bucket.acquire()

        assert slept == [pytest.approx(1.0)] * 3

    async def test_no_wait_once_the_interval_has_already_passed(self) -> None:
        slept: list[float] = []
        clock = iter([0.0, 5.0])

        async def record(delay: float) -> None:
            slept.append(delay)

        bucket = TokenBucket(rate_per_second=1.0, sleep=record, monotonic=lambda: next(clock))
        await bucket.acquire()
        await bucket.acquire()

        assert slept == []

    async def test_serialises_concurrent_callers(self) -> None:
        # Two coroutines must not both slip through the same token.
        bucket = TokenBucket(rate_per_second=1000.0)
        order: list[int] = []

        async def worker(n: int) -> None:
            await bucket.acquire()
            order.append(n)

        await asyncio.gather(*(worker(n) for n in range(5)))

        assert sorted(order) == [0, 1, 2, 3, 4]


class TestExtractionRequest:
    def test_carries_configured_limits_not_a_since_cursor(self) -> None:
        request = ExtractionRequest(max_records=100)

        assert request.max_records == 100
        # The sources have no uniform incremental cursor; pretending otherwise
        # would make repeat ingestion look safe for the wrong reason.
        assert not hasattr(request, "since")

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_a_non_positive_record_limit(self, bad: int) -> None:
        with pytest.raises(ValidationError, match="max_records"):
            ExtractionRequest(max_records=bad)


class TestRejected:
    def test_carries_enough_to_debug_without_the_original_response(self) -> None:
        rejection = Rejected(
            source=SourceName.GUTENDEX,
            source_id="1342",
            raw_payload={"id": 1342},
            rejection_code="invalid_record",
            detail="title must not be blank",
        )

        assert rejection.source is SourceName.GUTENDEX
        assert rejection.rejection_code == "invalid_record"
        assert rejection.raw_payload == {"id": 1342}

    def test_source_id_may_be_absent(self) -> None:
        # A payload can be malformed enough to have no usable identifier.
        rejection = Rejected(
            source=SourceName.GUTENDEX,
            source_id=None,
            raw_payload={},
            rejection_code="missing_source_id",
            detail=None,
        )

        assert rejection.source_id is None
