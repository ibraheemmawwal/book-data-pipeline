"""Configuration loading.

Configuration is environment-driven with no secret defaults. Google Books is
key-gated: absent credentials must yield an observable skip, never a crash and
never a silent no-op.
"""

import os

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from pipeline.config import MAX_GOODREADS_REQUESTS_PER_SECOND, Settings, SourceName
from pipeline.extract.goodreads import BLOCK_BACKOFF_FACTOR, BLOCK_BACKOFF_SECONDS

# Airflow's ``scheduler.task_instance_heartbeat_timeout`` default. A task that
# goes quiet for longer is treated as dead and killed, so it is a hard ceiling
# on any single wait the pipeline takes inside a task.
AIRFLOW_HEARTBEAT_TIMEOUT_SECONDS = 300.0


@pytest.fixture(autouse=True)
def _clear_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real .env leaking into assertions about defaults."""
    for key in list(os.environ):
        if key.startswith("PIPELINE_"):
            monkeypatch.delenv(key, raising=False)


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+psycopg://u:p@localhost:5432/catalogue",
        "openlibrary_contact_email": "owner@example.com",
    }
    return Settings(**(base | overrides))  # type: ignore[arg-type]


class TestDefaults:
    def test_load_batch_size_matches_the_trd(self) -> None:
        assert settings().load_batch_size == 1000

    def test_openlibrary_is_rate_limited_to_one_request_per_second(self) -> None:
        assert settings().openlibrary_requests_per_second == 1.0

    def test_kafka_topics_have_three_partitions(self) -> None:
        assert settings().kafka_topic_partitions == 3

    def test_processing_attempts_before_dlq_is_three(self) -> None:
        assert settings().kafka_max_processing_attempts == 3


class TestValidation:
    def test_database_url_is_required(self) -> None:
        with pytest.raises(ValidationError, match="database_url"):
            Settings(openlibrary_contact_email="owner@example.com")  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", ["sqlite:///local.db", "mysql://localhost/x", "not-a-url"])
    def test_non_postgresql_database_url_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="database_url"):
            settings(database_url=bad)

    def test_load_batch_size_is_capped_at_the_transaction_limit(self) -> None:
        # The TRD caps a load transaction at 1,000 records.
        with pytest.raises(ValidationError, match="load_batch_size"):
            settings(load_batch_size=1001)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_load_batch_size_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValidationError, match="load_batch_size"):
            settings(load_batch_size=bad)

    def test_openlibrary_rate_limit_above_one_per_second_is_rejected(self) -> None:
        # Politeness cap is a usage-policy commitment, not a tuning knob.
        with pytest.raises(ValidationError, match="openlibrary_requests_per_second"):
            settings(openlibrary_requests_per_second=5.0)

    def test_enabled_openlibrary_requires_a_contact_email(self) -> None:
        # Identified requests are required by the source usage policy.
        with pytest.raises(ValidationError, match="openlibrary_contact_email"):
            Settings(  # type: ignore[call-arg]
                database_url="postgresql+psycopg://u:p@localhost:5432/catalogue",
                openlibrary_enabled=True,
            )

    def test_whitespace_openlibrary_contact_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="openlibrary_contact_email"):
            Settings(  # type: ignore[call-arg]
                database_url="postgresql+psycopg://u:p@localhost:5432/catalogue",
                openlibrary_enabled=True,
                openlibrary_contact_email="   ",
            )

    def test_disabled_openlibrary_does_not_require_a_contact_email(self) -> None:
        loaded = Settings(  # type: ignore[call-arg]
            database_url="postgresql+psycopg://u:p@localhost:5432/catalogue",
            openlibrary_enabled=False,
        )

        assert loaded.openlibrary_contact_email is None


class TestUserAgent:
    def test_identifies_the_application_and_contact(self) -> None:
        agent = settings().user_agent()

        assert "book-data-pipeline" in agent
        assert "owner@example.com" in agent

    def test_omits_contact_when_openlibrary_is_disabled(self) -> None:
        agent = settings(openlibrary_enabled=False, openlibrary_contact_email=None).user_agent()

        assert "book-data-pipeline" in agent
        assert "@" not in agent


class TestActiveSources:
    def test_all_sources_active_when_configured(self) -> None:
        active = settings(
            googlebooks_api_key="test-key",
            goodreads_enabled=True,
            goodreads_unofficial_source_accepted=True,
        ).active_sources()

        # Resolution order, not the SourceName declaration order: Goodreads
        # resolves first, documented APIs fill gaps, Gutendex is last resort.
        assert active == (
            SourceName.GOODREADS,
            SourceName.OPENLIBRARY,
            SourceName.GOOGLEBOOKS,
            SourceName.GUTENDEX,
        )

    def test_goodreads_is_off_unless_both_gates_are_set(self) -> None:
        # Reading an unofficial contract must never be a default.
        assert SourceName.GOODREADS not in settings().active_sources()
        assert SourceName.GOODREADS not in settings(goodreads_enabled=True).active_sources()

    def test_enabling_without_accepting_the_risk_says_so(self) -> None:
        reason = settings(goodreads_enabled=True).skip_reason(SourceName.GOODREADS)

        assert reason is not None
        assert "risk has not been accepted" in reason

    def test_googlebooks_is_skipped_without_a_key(self) -> None:
        loaded = settings(googlebooks_enabled=True, googlebooks_api_key=None)

        assert SourceName.GOOGLEBOOKS not in loaded.active_sources()

    def test_missing_googlebooks_key_yields_a_reportable_skip_reason(self) -> None:
        # The reason is written to source_runs, not swallowed.
        loaded = settings(googlebooks_enabled=True, googlebooks_api_key=None)
        reason = loaded.skip_reason(SourceName.GOOGLEBOOKS)

        assert reason is not None
        assert "api key" in reason.lower()

    def test_active_source_has_no_skip_reason(self) -> None:
        assert settings().skip_reason(SourceName.GUTENDEX) is None

    def test_disabled_source_reports_being_disabled(self) -> None:
        reason = settings(gutendex_enabled=False).skip_reason(SourceName.GUTENDEX)

        assert reason is not None
        assert "disabled" in reason.lower()

    def test_at_least_one_source_must_be_active(self) -> None:
        with pytest.raises(ValidationError):
            settings(gutendex_enabled=False, openlibrary_enabled=False, googlebooks_enabled=False)

    def test_keyless_google_cannot_be_the_only_enabled_source(self) -> None:
        with pytest.raises(ValidationError, match="runnable"):
            settings(
                gutendex_enabled=False,
                openlibrary_enabled=False,
                googlebooks_enabled=True,
                googlebooks_api_key=None,
            )


class TestEnvironmentLoading:
    def test_reads_the_prefixed_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PIPELINE_DATABASE_URL", "postgresql+psycopg://u:p@db:5432/catalogue")
        monkeypatch.setenv("PIPELINE_OPENLIBRARY_CONTACT_EMAIL", "env@example.com")
        monkeypatch.setenv("PIPELINE_LOAD_BATCH_SIZE", "250")

        loaded = Settings()  # type: ignore[call-arg]

        assert loaded.load_batch_size == 250
        assert loaded.openlibrary_contact_email == "env@example.com"

    def test_unknown_pipeline_variables_are_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A typo'd variable that silently does nothing is a debugging trap.
        monkeypatch.setenv("PIPELINE_DATABASE_URL", "postgresql+psycopg://u:p@db:5432/catalogue")
        monkeypatch.setenv("PIPELINE_OPENLIBRARY_CONTACT_EMAIL", "env@example.com")
        monkeypatch.setenv("PIPELINE_LOAD_BATCH_SIZ", "250")

        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]

    def test_secrets_are_not_exposed_by_repr(self) -> None:
        loaded = settings(googlebooks_api_key="super-secret-key")

        assert "super-secret-key" not in repr(loaded)


class TestBlankSecrets:
    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_a_blank_api_key_reads_as_absent(self, blank: str) -> None:
        # PIPELINE_GOOGLEBOOKS_API_KEY= in an env file means "not configured".
        # An empty string would read as present-but-invalid and turn a clean
        # skip into a 400 from Google.
        loaded = settings(googlebooks_api_key=blank)

        assert loaded.googlebooks_api_key is None
        assert SourceName.GOOGLEBOOKS not in loaded.active_sources()


class TestDiscoverySettings:
    def test_the_language_filter_is_parsed_from_a_comma_list(self) -> None:
        assert settings(discovery_languages="eng, fra ,deu").discovery_language_set() == (
            frozenset({"eng", "fra", "deu"})
        )

    def test_an_empty_language_list_means_no_filter(self) -> None:
        # None rather than an empty set: an empty set would match nothing and
        # silently discover zero candidates.
        assert settings(discovery_languages="").discovery_language_set() is None

    def test_a_pinned_digest_is_normalised(self) -> None:
        loaded = settings(openlibrary_dump_sha256="A" * 64)

        assert loaded.openlibrary_dump_sha256 == "a" * 64

    @pytest.mark.parametrize("bad", ["abc", "z" * 64, "a" * 63, "a" * 65])
    def test_a_malformed_digest_is_refused_at_startup(self, bad: str) -> None:
        # Discovering for an hour and then failing verification is a much worse
        # way to learn someone pasted a truncated hash.
        with pytest.raises(ValidationError, match="openlibrary_dump_sha256"):
            settings(openlibrary_dump_sha256=bad)

    def test_an_absent_digest_is_allowed(self) -> None:
        assert settings().openlibrary_dump_sha256 is None


class TestThePlaceholderContact:
    """A contact address that reaches nobody.

    Open Library's guidance asks to be told who is calling, and the whole basis
    on which this pipeline uses it is that it identifies itself. A placeholder
    satisfies the letter of that and none of the point — and it had been going
    out on every request for a week without anything saying so.
    """

    @staticmethod
    def _settings(contact: str) -> Settings:
        return Settings(  # type: ignore[call-arg]
            database_url="postgresql+psycopg://u:p@localhost/db",
            openlibrary_contact_email=contact,
        )

    def test_it_warns(self) -> None:
        with capture_logs() as logs:
            self._settings("you@example.com").user_agent()

        assert any(entry["event"] == "config.placeholder_contact_email" for entry in logs)

    def test_a_real_address_is_silent(self) -> None:
        with capture_logs() as logs:
            self._settings("someone@somewhere.org").user_agent()

        assert not any(entry["event"] == "config.placeholder_contact_email" for entry in logs)

    def test_the_address_is_still_sent(self) -> None:
        """Not stripped, deliberately.

        Removing it would hide the misconfiguration from the one party in a
        position to notice it.
        """
        assert "you@example.com" in self._settings("you@example.com").user_agent()


class TestTheGoodreadsRate:
    """One request every five seconds, and why that number.

    The ceiling was always documented as a ceiling — "a lower observed safe
    rate wins" — and the first observation, taken while a block was in
    progress, walked the spacing out and read as a rolling budget of about
    eight requests: blocked at 2s, 5s and 15s, clean only at 30s.

    Re-run days later it does not reproduce. Sixteen consecutive requests at
    five seconds returned no block at all, and sixteen at thirty returned no
    block either; the only difference between the two sets is the 503 rate, 6
    of 16 against 4 of 16, which at that sample size is not a difference.

    So the block is episodic rather than a standing budget, and the machinery
    that survives one — escalating waits, the breaker, the cross-run cooldown —
    is what makes the faster spacing safe to hold rather than the block being
    gone.
    """

    @staticmethod
    def _settings() -> Settings:
        return Settings(  # type: ignore[call-arg]
            database_url="postgresql+psycopg://u:p@localhost/db",
            openlibrary_contact_email="t@example.com",
        )

    def test_the_default_is_the_observed_safe_rate(self) -> None:
        assert self._settings().goodreads_requests_per_second == pytest.approx(1.0 / 5.0)

    def test_it_stays_below_the_ceiling(self) -> None:
        settings = self._settings()

        assert settings.goodreads_requests_per_second <= MAX_GOODREADS_REQUESTS_PER_SECOND

    def test_the_ceiling_still_refuses_anything_faster(self) -> None:
        # The ceiling is what stops a well-meaning override becoming a crawl.
        with pytest.raises(ValidationError):
            Settings(  # type: ignore[call-arg]
                database_url="postgresql+psycopg://u:p@localhost/db",
                openlibrary_contact_email="t@example.com",
                goodreads_requests_per_second=MAX_GOODREADS_REQUESTS_PER_SECOND + 1,
            )

    def test_no_single_block_wait_outlives_an_airflow_heartbeat(self) -> None:
        """A wait long enough to look dead gets the task killed.

        Airflow fails a task that stops heartbeating for
        ``scheduler.task_instance_heartbeat_timeout`` — 300 seconds by default.
        A five-minute block wait sat exactly on it and was killed by it, so the
        mechanism for surviving a block was what ended the run.

        Patience comes from the number of waits now, not the length of one.
        """
        settings = self._settings()

        assert settings.goodreads_block_pause_seconds < AIRFLOW_HEARTBEAT_TIMEOUT_SECONDS

    def test_the_waits_still_outlast_a_measured_block(self) -> None:
        # Cutting the ceiling must not cut the total: a block took four to five
        # minutes to lift, and the ladder has to survive one.
        settings = self._settings()
        ladder = [
            min(
                settings.goodreads_block_pause_seconds,
                BLOCK_BACKOFF_SECONDS * BLOCK_BACKOFF_FACTOR ** (attempt - 1),
            )
            for attempt in range(1, settings.goodreads_block_retries + 1)
        ]

        assert ladder == [10.0, 60.0, 120.0, 120.0, 120.0]
        assert sum(ladder) > 300.0

    def test_only_one_request_may_be_in_flight(self) -> None:
        # Pacing and concurrency are separate promises; slowing down must not
        # quietly permit two at once.
        assert self._settings().goodreads_max_in_flight == 1
