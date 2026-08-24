"""Chaos scenarios: the experiments that produce the README's numbers.

Each scenario is a controlled A/B. On one side, a naive client -- what you get
if you point `httpx` at a provider and add a retry loop, which is what most
applications actually do. On the other, the gateway.

Both face an identical provider with an identical failure profile and an
identical random seed, so any difference in outcome is attributable to the
reliability layer and nothing else. Without that control the numbers would be
marketing, not measurement.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from gateway.providers.mock import MockBehaviour, MockProvider
from gateway.reliability.budget import RetryBudget
from gateway.reliability.executor import ReliableExecutor
from gateway.types import ChatRequest, Message, ProviderError


def make_request(i: int) -> ChatRequest:
    return ChatRequest(messages=(Message(role="user", content=f"request {i}"),))


# --- Result collection -----------------------------------------------------


@dataclass(slots=True)
class Outcome:
    """Everything measured about a single benchmark run."""

    label: str
    latencies_ms: list[float] = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    upstream_calls: int = 0
    """Total requests that actually reached a provider.

    This is the cost axis. A reliability mechanism that doubles upstream calls
    to shave 5ms off p99 is a bad trade, and reporting only latency would hide
    that.
    """

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def pct(self, p: float) -> float:
        """Percentile latency over SUCCESSFUL requests only.

        Including failures would let a fast-failing system look fast, which is
        exactly backwards.
        """
        if not self.latencies_ms:
            return float("nan")
        ordered = sorted(self.latencies_ms)
        idx = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[idx]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else float("nan")


async def _drive(
    label: str,
    call: Callable[[int], Awaitable[object]],
    *,
    n: int,
    concurrency: int,
) -> Outcome:
    """Fire `n` requests with bounded concurrency and record what happened.

    A semaphore bounds in-flight work rather than releasing all `n` at once.
    Unbounded concurrency would measure how fast the benchmark can exhaust
    memory, not how the gateway behaves under steady load.
    """
    outcome = Outcome(label=label)
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> None:
        async with sem:
            started = time.perf_counter()
            try:
                await call(i)
            except Exception:  # noqa: BLE001 - any failure is a failure here
                outcome.failures += 1
            else:
                outcome.successes += 1
                outcome.latencies_ms.append((time.perf_counter() - started) * 1000)

    await asyncio.gather(*(one(i) for i in range(n)))
    return outcome


# --- The naive baseline ----------------------------------------------------


class NaiveClient:
    """What most applications actually do: one provider, a fixed retry loop.

    Deliberately not a straw man -- it does retry, and it does back off. It
    simply lacks the system-level mechanisms: no breaker, no budget, no
    hedging, no failover.
    """

    def __init__(self, provider: MockProvider, *, max_retries: int = 2) -> None:
        self.provider = provider
        self.max_retries = max_retries

    async def complete(self, request: ChatRequest) -> object:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self.provider.complete(request)
            except ProviderError as exc:
                last = exc
                await asyncio.sleep(0.05 * (2**attempt))
        raise last if last else RuntimeError("unreachable")

    def caller(self) -> Callable[[int], Awaitable[object]]:
        """Adapt to the `call(i)` signature `_drive` expects."""
        return lambda i: self.complete(make_request(i))


# --- Scenarios -------------------------------------------------------------


async def scenario_degraded_provider(n: int = 400, concurrency: int = 40) -> list[Outcome]:
    """Can the gateway stay up when the primary provider is failing hard?

    The headline experiment. A provider dropping ~30% of requests is a bad but
    entirely realistic day.
    """
    profile = MockBehaviour(error_rate=0.30, tail_probability=0.05, seed=7)

    naive = NaiveClient(MockProvider("degraded", profile))
    naive_out = await _drive("naive (retry only)", naive.caller(), n=n, concurrency=concurrency)

    primary = MockProvider("degraded", profile)
    backup = MockProvider("backup", MockBehaviour(seed=8))
    gw = ReliableExecutor(
        [primary, backup],
        hedge_delay_s=0.5,
        max_retries=2,
        retry_budget=RetryBudget(ratio=0.2),
    )
    gw_out = await _drive(
        "gateway",
        lambda i: gw.complete(make_request(i)),
        n=n,
        concurrency=concurrency,
    )
    gw_out.upstream_calls = primary.call_count + backup.call_count
    naive_out.upstream_calls = naive.provider.call_count
    return [naive_out, gw_out]


async def scenario_tail_latency(n: int = 400, concurrency: int = 40) -> list[Outcome]:
    """Does hedging actually cut the tail, and what does it cost?

    Both sides use the SAME provider profile -- a heavy tail, no errors. The
    only difference is whether a hedge is allowed to fire, which isolates the
    mechanism.
    """
    profile = MockBehaviour(base_latency_s=0.05, tail_latency_s=2.0, tail_probability=0.10, seed=11)

    # Hedging disabled: an enormous delay means the hedge deadline never hits.
    p_off = MockProvider("primary", profile)
    backup_profile = MockBehaviour(base_latency_s=0.05, tail_probability=0.0, seed=12)
    b_off = MockProvider("backup", backup_profile)
    off = ReliableExecutor([p_off, b_off], hedge_delay_s=1e6, max_retries=0)
    off_out = await _drive(
        "hedging off", lambda i: off.complete(make_request(i)), n=n, concurrency=concurrency
    )
    off_out.upstream_calls = p_off.call_count + b_off.call_count

    # Hedging enabled at roughly the healthy p95.
    p_on = MockProvider("primary", profile)
    b_on = MockProvider("backup", backup_profile)
    on = ReliableExecutor([p_on, b_on], hedge_delay_s=0.2, max_retries=0)
    on_out = await _drive(
        "hedging on", lambda i: on.complete(make_request(i)), n=n, concurrency=concurrency
    )
    on_out.upstream_calls = p_on.call_count + b_on.call_count
    on_out.label = f"hedging on ({on.hedges_fired} fired, {on.hedges_won} won)"
    return [off_out, on_out]


async def scenario_retry_storm(n: int = 300, concurrency: int = 50) -> list[Outcome]:
    """During a total outage, how much load does each approach inflict upstream?

    Nobody succeeds here -- the provider is completely down. That is the point.
    The question is not "who stays up" but "who makes the outage worse", and
    the measurement that matters is `upstream_calls`.
    """
    down = MockBehaviour(error_rate=1.0, base_latency_s=0.01, tail_probability=0.0, seed=13)

    naive_provider = MockProvider("down", down)
    naive = NaiveClient(naive_provider, max_retries=2)
    naive_out = await _drive("naive (retry only)", naive.caller(), n=n, concurrency=concurrency)
    naive_out.upstream_calls = naive_provider.call_count

    gw_provider = MockProvider("down", down)
    gw = ReliableExecutor(
        [gw_provider],
        hedge_delay_s=1e6,
        max_retries=2,
        retry_budget=RetryBudget(ratio=0.15, min_allowance=3),
        breaker_failure_threshold=5,
        breaker_recovery_seconds=30.0,
    )
    gw_out = await _drive(
        "gateway (budget + breaker)",
        lambda i: gw.complete(make_request(i)),
        n=n,
        concurrency=concurrency,
    )
    gw_out.upstream_calls = gw_provider.call_count
    return [naive_out, gw_out]
