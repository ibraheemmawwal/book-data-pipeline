"""Unit-test guards.

Unit tests must not touch the network. Several resolver tests silently did:
they were slow, they would be flaky in CI, and they sent real traffic to Open
Library and Goodreads every time anyone ran the suite. Catching that by reading
the code clearly did not work, so it is enforced here instead.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import respx


@pytest.fixture(autouse=True)
def _no_unmocked_network() -> Iterator[None]:
    """Fail any unit test that makes a real HTTP request.

    ``assert_all_mocked`` turns an unmocked call into an immediate error naming
    the URL, which is a far better failure than a slow test that passes until
    the day the source is down.
    """
    # The bare global router, not respx.mock(...): a configured router is a
    # separate instance, and module-level respx.get(...) calls in tests would
    # register on the global one while this fixture activated a different
    # empty one — so every request would look unmocked.
    with respx.mock:
        yield
