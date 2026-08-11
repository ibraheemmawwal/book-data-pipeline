"""Configuration loading.

Configuration is environment-driven with no secret defaults. Google Books is
key-gated: absent credentials must yield an observable skip, never a crash and
never a silent no-op.
"""

import os

import pytest
from pydantic import ValidationError

from pipeline.config import Settings, SourceName


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
        active = settings(googlebooks_api_key="test-key").active_sources()

        assert active == (SourceName.GUTENDEX, SourceName.OPENLIBRARY, SourceName.GOOGLEBOOKS)

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
