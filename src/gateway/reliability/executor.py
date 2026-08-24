"""The routing core: retries, failover, and hedged requests.

This module is the reason the project exists. Everything else is plumbing.

Three mechanisms, each fixing a different failure mode:

  retry      fixes a request that failed
  failover   fixes a provider that failed
  hedging    fixes a request that is merely SLOW -- which no amount of
             retrying will ever help, because nothing has failed yet

Hedging is the interesting one. Tail latency in distributed systems is usually
not caused by a slow *service* but by a slow *instance*: a cold cache, a noisy
neighbour, an unlucky GC pause. The fix is to stop waiting and ask someone
else, while keeping the original in flight in case it lands first. The cost is
bounded: fire the hedge at p95, and you add duplicate load to only ~5% of
requests in exchange for cutting the tail dramatically.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Sequence

import structlog

from gateway.providers.base import Provider
from gateway.reliability.breaker import CircuitBreaker
from gateway.reliability.budget import RetryBudget
from gateway.types import (
    ChatRequest,
    ChatResponse,
    GatewayError,
    NoProviderAvailableError,
    PermanentProviderError,
    ProviderError,
    RateLimitError,
    StreamChunk,
    TransientProviderError,
)

log = structlog.get_logger(__name__)


class ReliableExecutor:
    """Executes a request across a pool of providers, as reliably as possible."""

    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        hedge_delay_s: float = 0.8,
        max_retries: int = 2,
        retry_budget: RetryBudget | None = None,
        breaker_failure_threshold: int = 5,
        breaker_recovery_seconds: float = 15.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        if not providers:
            raise ValueError("ReliableExecutor requires at least one provider")

        self.providers = list(providers)
        self.hedge_delay_s = hedge_delay_s
        self.max_retries = max_retries
        self.request_timeout_s = request_timeout_s
        self.budget = retry_budget or RetryBudget()
        self.breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(
                p.name,
                failure_threshold=breaker_failure_threshold,
                recovery_seconds=breaker_recovery_seconds,
            )
            for p in self.providers
        }
        self.hedges_fired = 0
        self.hedges_won = 0

    # -- provider selection -------------------------------------------------

    async def _healthy_providers(self) -> list[Provider]:
        """Providers whose breaker will currently admit a request.

        Order is preserved: providers[0] is the primary, the rest are failover
        targets in priority order.
        """
        healthy: list[Provider] = []
        for p in self.providers:
            if await self.breakers[p.name].allows_request():
                healthy.append(p)
        return healthy

    # -- single attempt -----------------------------------------------------

    async def _attempt(self, provider: Provider, request: ChatRequest) -> ChatResponse:
        """One call to one provider, with breaker bookkeeping and a timeout."""
        breaker = self.breakers[provider.name]
        try:
            response = await asyncio.wait_for(
                provider.complete(request), timeout=self.request_timeout_s
            )
        except TimeoutError as exc:
            # A timeout is indistinguishable from a 503 as far as the caller is
            # concerned, so normalise it into the same taxonomy rather than
            # letting asyncio's exception type leak upward.
            await breaker.record_failure()
            raise TransientProviderError(
                f"timed out after {self.request_timeout_s}s", provider=provider.name
            ) from exc
        except PermanentProviderError:
            # A malformed request is our fault, not the provider's. Counting it
            # against the breaker would let one bad client take a healthy
            # provider offline for everyone.
            raise
        except ProviderError:
            await breaker.record_failure()
            raise

        await breaker.record_success()
        self.budget.record_success()
        return response

    # -- hedging ------------------------------------------------------------

    async def _hedged(self, providers: list[Provider], request: ChatRequest) -> ChatResponse:
        """Race the primary against delayed backups; return the first success.

        The delay is what separates hedging from plain duplication. Fire both
        immediately and you double your bill on every single request. Fire the
        second only after the first has already proven slow, and you pay the
        duplicate cost on just the tail.
        """
        primary, *backups = providers
        pending: set[asyncio.Task[ChatResponse]] = {
            asyncio.create_task(self._attempt(primary, request), name=primary.name)
        }
        errors: list[BaseException] = []

        try:
            while pending or backups:
                if not pending:
                    # Everything in flight has failed. This is failover, not
                    # hedging: there is no point waiting out the hedge delay
                    # when we already know the answer is not coming.
                    nxt = backups.pop(0)
                    log.info("failover.next", provider=nxt.name)
                    pending.add(asyncio.create_task(self._attempt(nxt, request), name=nxt.name))

                # Wait only until the hedge deadline. Two things can happen:
                # something finishes (handle it), or nothing does (the request
                # is in the slow tail -- add a competitor). With no backups
                # left there is nothing to hedge with, so wait indefinitely.
                timeout = self.hedge_delay_s if backups else None
                done, pending = await asyncio.wait(
                    pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    exc = task.exception()
                    if exc is None:
                        if task.get_name() != primary.name:
                            self.hedges_won += 1
                        return task.result()
                    if isinstance(exc, PermanentProviderError):
                        # The request itself is malformed. Every other provider
                        # will reject it too, so fail fast and surface the real
                        # error instead of burying it under "all providers down".
                        raise exc
                    errors.append(exc)

                if not done and backups:
                    backup = backups.pop(0)
                    self.hedges_fired += 1
                    log.info("hedge.fired", primary=primary.name, backup=backup.name)
                    pending.add(
                        asyncio.create_task(self._attempt(backup, request), name=backup.name)
                    )

            raise NoProviderAvailableError(
                f"all attempts failed: {[type(e).__name__ for e in errors]}"
            )
        finally:
            # Whether we won, lost, or raised, every still-running duplicate
            # must be cancelled -- otherwise a hedge that loses the race keeps
            # burning tokens nobody will ever read, and the task leaks.
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    # -- public API ---------------------------------------------------------

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Execute a request with the full reliability stack applied."""
        last_error: BaseException | None = None

        for attempt in range(self.max_retries + 1):
            providers = await self._healthy_providers()
            if not providers:
                raise NoProviderAvailableError("every provider circuit is open")

            if attempt > 0:
                if not self.budget.try_consume():
                    log.warning("retry.budget_exhausted", **self.budget.stats)
                    break
                await asyncio.sleep(self._backoff(attempt, last_error))

            try:
                return await self._hedged(providers, request)
            except PermanentProviderError:
                # Retrying a 400 just produces another 400, more slowly.
                raise
            except (GatewayError, ProviderError) as exc:
                last_error = exc
                log.warning("attempt.failed", attempt=attempt, error=str(exc))

        raise NoProviderAvailableError(f"exhausted retries; last error: {last_error}")

    def _backoff(self, attempt: int, last_error: BaseException | None) -> float:
        """Exponential backoff with full jitter.

        Jitter is not a nicety. Without it, every client that failed at the
        same moment retries at the same moment, reproducing the original
        traffic spike at each backoff step. Randomising spreads the retries
        across the window -- this is AWS's "full jitter" strategy.
        """
        if isinstance(last_error, RateLimitError) and last_error.retry_after:
            return last_error.retry_after
        ceiling = min(2.0**attempt * 0.1, 5.0)
        return random.uniform(0, ceiling)

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamChunk]:
        """Stream a response, failing over only BEFORE the first byte.

        This is the subtle part. Failover is safe right up until the client has
        seen output; after that, switching providers would splice two different
        answers together and hand the user a corrupted response.

        So: try each provider in turn, but the moment the first chunk is
        yielded, commit. From then on an error propagates to the client instead
        of triggering a failover.
        """
        providers = await self._healthy_providers()
        if not providers:
            raise NoProviderAvailableError("every provider circuit is open")

        last_error: BaseException | None = None

        for provider in providers:
            breaker = self.breakers[provider.name]
            committed = False
            try:
                async for chunk in provider.stream(request):
                    if not committed:
                        committed = True
                        await breaker.record_success()
                        self.budget.record_success()
                    yield chunk
                return
            except (ProviderError, TimeoutError) as exc:
                if committed:
                    log.error("stream.failed_after_commit", provider=provider.name)
                    raise
                await breaker.record_failure()
                last_error = exc
                log.warning("stream.failover", provider=provider.name, error=str(exc))
                continue

        raise NoProviderAvailableError(f"no provider could start a stream; last: {last_error}")

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()
