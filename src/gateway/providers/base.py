"""The provider interface every backend implements.

Two methods, deliberately. `complete()` returns a whole response; `stream()`
yields chunks. They are separate rather than one method with a `stream` flag
because their failure semantics differ fundamentally:

  - `complete()` is atomic. It either produced an answer or it did not, so the
    reliability layer is free to retry it or race two of them.
  - `stream()` is NOT atomic. Once the first byte reaches the client, the
    request is committed -- you cannot transparently retry, because the client
    has already seen half an answer.

Nearly every naive gateway gets this wrong. See `reliability/executor.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from gateway.types import ChatRequest, ChatResponse, StreamChunk


class Provider(ABC):
    """An upstream LLM backend."""

    name: str
    """Stable identifier used in metrics, logs, and the cost ledger."""

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Run a request to completion.

        Raises:
            TransientProviderError: retryable failure (5xx, timeout).
            RateLimitError: retryable, but back off first.
            PermanentProviderError: will never succeed as-is.
        """

    @abstractmethod
    def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Yield the response incrementally.

        Note this is `def`, not `async def`: an async generator function is
        already a callable returning an AsyncIterator. Declaring it `async def`
        would make callers await a coroutine that returns a generator -- an
        easy and very common mistake.
        """

    async def aclose(self) -> None:
        """Release connection pools. Called on application shutdown."""
        return None

    def __repr__(self) -> str:
        return f"<Provider {self.name}>"
