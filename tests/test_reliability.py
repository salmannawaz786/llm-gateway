"""Tests for the reliability layer.

These are the tests that matter. Anyone can test that a happy-path request
returns 200; the interesting question is what happens when providers misbehave.
Each test below corresponds to a specific production failure mode.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from gateway.providers.mock import MockBehaviour, MockProvider
from gateway.reliability.breaker import BreakerState, CircuitBreaker
from gateway.reliability.budget import RetryBudget
from gateway.reliability.executor import ReliableExecutor
from gateway.types import (
    ChatRequest,
    ChunkType,
    Message,
    NoProviderAvailableError,
    PermanentProviderError,
)


def make_request(text: str = "hello") -> ChatRequest:
    return ChatRequest(messages=(Message(role="user", content=text),))


# --- Circuit breaker -------------------------------------------------------


async def test_breaker_opens_after_threshold() -> None:
    breaker = CircuitBreaker("p", failure_threshold=3, recovery_seconds=60)

    for _ in range(3):
        assert await breaker.allows_request()
        await breaker.record_failure()

    assert breaker.state is BreakerState.OPEN
    # The point of the breaker: subsequent calls fail instantly instead of
    # each paying a full timeout against a provider we know is dead.
    assert not await breaker.allows_request()


async def test_breaker_half_open_admits_exactly_one_probe() -> None:
    breaker = CircuitBreaker("p", failure_threshold=1, recovery_seconds=0.05)
    await breaker.record_failure()
    assert breaker.state is BreakerState.OPEN

    await asyncio.sleep(0.06)

    # Ten concurrent callers, but only one may become the trial request --
    # otherwise a recovering provider gets the whole backlog at once.
    results = await asyncio.gather(*(breaker.allows_request() for _ in range(10)))
    assert sum(results) == 1
    assert breaker.state.value == BreakerState.HALF_OPEN.value


async def test_breaker_closes_after_successful_probe() -> None:
    breaker = CircuitBreaker("p", failure_threshold=1, recovery_seconds=0.01)
    await breaker.record_failure()
    await asyncio.sleep(0.02)
    assert await breaker.allows_request()
    await breaker.record_success()
    assert breaker.state is BreakerState.CLOSED


async def test_breaker_reopens_if_probe_fails() -> None:
    breaker = CircuitBreaker("p", failure_threshold=5, recovery_seconds=0.01)
    await breaker.record_failure()
    breaker._state = BreakerState.HALF_OPEN  # noqa: SLF001
    await breaker.record_failure()
    # A failed probe must reopen immediately, without waiting to re-accumulate
    # `failure_threshold` failures -- we already have our answer.
    assert breaker.state is BreakerState.OPEN


# --- Retry budget ----------------------------------------------------------


def test_budget_blocks_retry_storm() -> None:
    """The core property: retries are capped by recent SUCCESS volume.

    During a total outage there are no successes, so the budget collapses to
    its floor and retries stop -- instead of tripling load on a provider that
    is already failing.
    """
    budget = RetryBudget(ratio=0.1, min_allowance=2)

    for _ in range(100):
        budget.record_success()

    allowed = sum(budget.try_consume() for _ in range(100))
    assert allowed == 10  # 10% of 100 successes

    exhausted = RetryBudget(ratio=0.1, min_allowance=2)
    assert sum(exhausted.try_consume() for _ in range(100)) == 2  # floor only


@given(
    successes=st.integers(min_value=0, max_value=500),
    ratio=st.floats(min_value=0.01, max_value=0.9),
)
@hyp_settings(max_examples=50, deadline=None)
def test_budget_never_exceeds_allowance(successes: int, ratio: float) -> None:
    """Property: no traffic pattern can ever exceed the computed allowance."""
    budget = RetryBudget(ratio=ratio, min_allowance=3)
    for _ in range(successes):
        budget.record_success()

    expected = max(3, int(successes * ratio))
    granted = sum(budget.try_consume() for _ in range(expected + 50))
    assert granted == expected


# --- Executor: failover ----------------------------------------------------


async def test_failover_to_healthy_provider() -> None:
    dead = MockProvider("dead", MockBehaviour.down())
    alive = MockProvider("alive", MockBehaviour.healthy())
    executor = ReliableExecutor([dead, alive], hedge_delay_s=0.01)

    response = await executor.complete(make_request())
    assert response.provider == "alive"


async def test_permanent_errors_are_not_retried() -> None:
    """A 400 will fail identically forever. Retrying just wastes time."""
    bad = MockProvider("bad", MockBehaviour(permanent_error_rate=1.0))
    executor = ReliableExecutor([bad], hedge_delay_s=0.01, max_retries=3)

    with pytest.raises(PermanentProviderError):
        await executor.complete(make_request())

    assert bad.call_count == 1


async def test_all_providers_down_raises() -> None:
    executor = ReliableExecutor(
        [MockProvider("a", MockBehaviour.down()), MockProvider("b", MockBehaviour.down())],
        hedge_delay_s=0.01,
        max_retries=1,
    )
    with pytest.raises(NoProviderAvailableError):
        await executor.complete(make_request())


# --- Executor: hedging -----------------------------------------------------


async def test_hedge_fires_only_when_primary_is_slow() -> None:
    slow = MockProvider("slow", MockBehaviour(base_latency_s=1.0, tail_probability=0.0))
    fast = MockProvider("fast", MockBehaviour(base_latency_s=0.01, tail_probability=0.0))
    executor = ReliableExecutor([slow, fast], hedge_delay_s=0.05)

    response = await executor.complete(make_request())

    # The hedge should have overtaken the slow primary.
    assert response.provider == "fast"
    assert executor.hedges_fired == 1


async def test_no_hedge_when_primary_is_fast() -> None:
    """Hedging must not fire on healthy traffic -- that would double the bill."""
    fast = MockProvider("fast", MockBehaviour(base_latency_s=0.01, tail_probability=0.0))
    backup = MockProvider("backup", MockBehaviour(base_latency_s=0.01, tail_probability=0.0))
    executor = ReliableExecutor([fast, backup], hedge_delay_s=0.5)

    await executor.complete(make_request())

    assert executor.hedges_fired == 0
    assert backup.call_count == 0


async def test_losing_hedge_is_cancelled() -> None:
    """A hedge that loses the race must be cancelled, not left running.

    Otherwise every hedged request keeps burning upstream tokens for a
    response nobody will ever read.
    """
    slow = MockProvider("slow", MockBehaviour(base_latency_s=5.0, tail_probability=0.0))
    fast = MockProvider("fast", MockBehaviour(base_latency_s=0.01, tail_probability=0.0))
    executor = ReliableExecutor([slow, fast], hedge_delay_s=0.02)

    before = len(asyncio.all_tasks())
    await executor.complete(make_request())
    await asyncio.sleep(0.05)

    # No orphaned task should survive the call.
    assert len(asyncio.all_tasks()) <= before


# --- Executor: streaming ---------------------------------------------------


async def test_stream_fails_over_before_first_chunk() -> None:
    dead = MockProvider("dead", MockBehaviour.down())
    alive = MockProvider("alive", MockBehaviour.healthy())
    executor = ReliableExecutor([dead, alive], hedge_delay_s=0.01)

    chunks = [c async for c in executor.stream(make_request())]

    assert any(c.type is ChunkType.DELTA for c in chunks)
    assert chunks[-1].type is ChunkType.DONE


async def test_stream_yields_deltas_then_done() -> None:
    executor = ReliableExecutor([MockProvider("p", MockBehaviour.healthy())])
    chunks = [c async for c in executor.stream(make_request())]

    assert chunks[-1].type is ChunkType.DONE
    assert chunks[-1].usage is not None
    text = "".join(c.delta for c in chunks if c.type is ChunkType.DELTA)
    assert text.strip()


# --- Concurrency -----------------------------------------------------------


async def test_handles_concurrent_load() -> None:
    """Sanity check that nothing in the executor serialises requests.

    100 requests against a provider with 50ms latency should finish in
    well under a second. If this takes 5 seconds, something is blocking the
    event loop.
    """
    executor = ReliableExecutor(
        [MockProvider("p", MockBehaviour(base_latency_s=0.05, tail_probability=0.0))],
        hedge_delay_s=10.0,
    )

    start = asyncio.get_running_loop().time()
    responses = await asyncio.gather(*(executor.complete(make_request()) for _ in range(100)))
    elapsed = asyncio.get_running_loop().time() - start

    assert len(responses) == 100
    assert elapsed < 1.0
