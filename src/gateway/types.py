"""Core domain types.

Everything in the gateway speaks these types. Provider adapters translate
between a vendor's wire format and these; nothing else in the codebase should
ever know what an "Anthropic message" or a "Gemini candidate" looks like.

That boundary is the whole reason a gateway is possible: add a provider by
writing one adapter, and every feature (caching, hedging, breakers, billing)
works with it for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A provider-agnostic completion request."""

    messages: tuple[Message, ...]
    model: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.7
    stream: bool = False

    def cache_key_text(self) -> str:
        """The text used for semantic cache lookups.

        Only the conversation is embedded -- sampling parameters are handled
        separately, because two requests with identical text but different
        temperatures are *not* interchangeable.
        """
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ChatResponse:
    """A completed, non-streaming response."""

    content: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    cached: bool = False
    cache_similarity: float | None = None


class ChunkType(StrEnum):
    DELTA = "delta"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """One incremental piece of a streaming response.

    `usage` is only populated on the final DONE chunk, because most providers
    only report token counts once the stream terminates.
    """

    type: ChunkType
    delta: str = ""
    usage: Usage | None = None


# --- Errors ----------------------------------------------------------------
# The retry/hedging logic branches on these, so the taxonomy matters more than
# it looks. Getting it wrong means either retrying something that will never
# succeed (wasted money) or giving up on something transient (lost uptime).


class GatewayError(Exception):
    """Base class for every error the gateway raises."""


class ProviderError(GatewayError):
    """Upstream provider failed."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class TransientProviderError(ProviderError):
    """A failure that may succeed if tried again: 5xx, timeout, connection reset."""

    retryable = True


class RateLimitError(TransientProviderError):
    """429 from upstream. Retryable, but only after backing off."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        retry_after: float | None = None,
        status_code: int | None = 429,
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code)
        self.retry_after = retry_after


class PermanentProviderError(ProviderError):
    """4xx that will fail identically forever: bad auth, malformed request."""

    retryable = False


class CircuitOpenError(GatewayError):
    """The breaker refused the call because the provider is known to be down."""


class NoProviderAvailableError(GatewayError):
    """Every candidate provider was unhealthy, exhausted, or failed."""
