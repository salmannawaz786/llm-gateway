"""A fully controllable fake provider.

This is not a testing afterthought -- it is the reason the reliability claims
in the README are provable. You cannot ask Groq to fail 30% of requests, or to
add 4 seconds of latency to its p95, so you cannot demonstrate a circuit
breaker or a hedged request against a real provider. Here you can.

It also means the whole project clones and runs with zero API keys.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass

from gateway.providers.base import Provider
from gateway.types import (
    ChatRequest,
    ChatResponse,
    ChunkType,
    PermanentProviderError,
    RateLimitError,
    StreamChunk,
    TransientProviderError,
    Usage,
)

_LOREM = (
    "The gateway routes the request, records the cost, and returns the answer. "
    "Reliability is measured, not assumed. Latency is a distribution, not a number."
)


@dataclass(slots=True)
class MockBehaviour:
    """The failure profile to simulate.

    Latency is drawn from a two-mode distribution rather than a flat range,
    because that is what real providers actually look like: most requests are
    fast, and a small tail is dramatically slower. A uniform distribution would
    make hedging look useless, since hedging only pays off against a heavy tail.
    """

    base_latency_s: float = 0.05
    tail_latency_s: float = 3.0
    tail_probability: float = 0.05

    error_rate: float = 0.0
    rate_limit_rate: float = 0.0
    permanent_error_rate: float = 0.0

    tokens_per_second: float = 200.0
    seed: int | None = None

    @classmethod
    def healthy(cls) -> MockBehaviour:
        return cls()

    @classmethod
    def degraded(cls) -> MockBehaviour:
        """A provider having a bad day: 30% failures and a fat latency tail."""
        return cls(error_rate=0.25, rate_limit_rate=0.05, tail_probability=0.25)

    @classmethod
    def down(cls) -> MockBehaviour:
        return cls(error_rate=1.0)


class MockProvider(Provider):
    def __init__(
        self,
        name: str = "mock",
        behaviour: MockBehaviour | None = None,
        *,
        model: str = "mock-1",
    ) -> None:
        self.name = name
        self.behaviour = behaviour or MockBehaviour.healthy()
        self.model = model
        # A dedicated Random instance keeps runs reproducible without touching
        # the global random state that the rest of the process may rely on.
        self._rng = random.Random(self.behaviour.seed)
        self.call_count = 0

    # -- internals ----------------------------------------------------------

    def _roll_failure(self) -> None:
        r = self._rng.random()
        if r < self.behaviour.permanent_error_rate:
            raise PermanentProviderError(
                "simulated bad request", provider=self.name, status_code=400
            )
        if r < self.behaviour.permanent_error_rate + self.behaviour.rate_limit_rate:
            raise RateLimitError("simulated rate limit", provider=self.name, retry_after=0.2)
        if (
            r
            < self.behaviour.permanent_error_rate
            + self.behaviour.rate_limit_rate
            + self.behaviour.error_rate
        ):
            raise TransientProviderError(
                "simulated upstream 503", provider=self.name, status_code=503
            )

    def _draw_latency(self) -> float:
        if self._rng.random() < self.behaviour.tail_probability:
            return self.behaviour.tail_latency_s
        return self.behaviour.base_latency_s

    def _answer_for(self, request: ChatRequest) -> str:
        last = request.messages[-1].content if request.messages else ""
        return f"[{self.name}] {_LOREM} (re: {last[:60]})"

    # -- Provider API -------------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.call_count += 1
        self._roll_failure()
        # asyncio.sleep is what makes this a *concurrency* simulation rather
        # than a blocking one: while this task sleeps, the event loop runs
        # every other in-flight request. time.sleep() here would freeze the
        # entire server and silently invalidate every benchmark.
        await asyncio.sleep(self._draw_latency())

        text = self._answer_for(request)
        return ChatResponse(
            content=text,
            model=request.model or self.model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=len(request.cache_key_text()) // 4,
                completion_tokens=len(text) // 4,
            ),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        self.call_count += 1
        self._roll_failure()
        await asyncio.sleep(self._draw_latency())

        text = self._answer_for(request)
        words = text.split(" ")
        delay = 1.0 / self.behaviour.tokens_per_second

        for word in words:
            await asyncio.sleep(delay)
            yield StreamChunk(type=ChunkType.DELTA, delta=word + " ")

        yield StreamChunk(
            type=ChunkType.DONE,
            usage=Usage(
                prompt_tokens=len(request.cache_key_text()) // 4,
                completion_tokens=len(words),
            ),
        )
