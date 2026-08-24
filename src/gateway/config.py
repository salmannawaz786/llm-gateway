"""Configuration, loaded from environment variables / `.env`.

Every tunable in the reliability layer is exposed here rather than hardcoded,
because the chaos benchmarks work by sweeping these values and measuring the
effect. A knob you cannot turn is a claim you cannot prove.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        extra="ignore",
    )

    log_level: str = "INFO"
    database_url: str = "sqlite:///./gateway.db"

    # --- Providers ---------------------------------------------------------
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"

    # --- Reliability -------------------------------------------------------
    request_timeout_s: float = 30.0

    hedge_delay_ms: int = 800
    """How long to wait before firing a duplicate request at a second provider.

    Set this near the p95 of a healthy provider. Too low and you double your
    spend for nothing; too high and slow requests stay slow.
    """

    retry_budget_ratio: float = 0.15
    """Retries are capped at this fraction of recent successful requests.

    Without a budget, a provider-wide outage turns every client request into
    N requests and the retries themselves become the outage. This is the single
    most important line of defence against a retry storm.
    """

    max_retries: int = 2

    breaker_failure_threshold: int = 5
    """Consecutive failures before a provider's circuit opens."""

    breaker_recovery_seconds: float = 15.0
    """How long the circuit stays open before allowing one trial request."""

    # --- Semantic cache ----------------------------------------------------
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.92
    """Cosine similarity above which a cached response is considered a hit.

    This is a precision/recall dial with real money on one side and real
    correctness on the other. See DESIGN.md for how the value was chosen.
    """

    cache_max_entries: int = 10_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so config is parsed once per process, not once per request."""
    return Settings()
