"""The gateway service: cache + single-flight in front of the executor.

Kept separate from `ReliableExecutor` on purpose. The executor's job is
"get this request answered despite failures"; caching is a different concern
and folding it in would make both harder to test. This layer composes them.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from gateway.cache.semantic import SemanticCache
from gateway.observability import metrics
from gateway.reliability.executor import ReliableExecutor
from gateway.types import ChatRequest, ChatResponse

log = structlog.get_logger(__name__)


class GatewayService:
    def __init__(
        self,
        executor: ReliableExecutor,
        cache: SemanticCache | None = None,
    ) -> None:
        self.executor = executor
        self.cache = cache
        # Single-flight: maps an exact request key to the in-flight task
        # already fetching it.
        self._inflight: dict[str, asyncio.Future[ChatResponse]] = {}

    @staticmethod
    def _exact_key(request: ChatRequest) -> str:
        return (
            f"{request.model or 'default'}|{request.temperature}"
            f"|{request.max_tokens}|{request.cache_key_text()}"
        )

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Answer a request, using the cache and deduplicating concurrent twins.

        Single-flight matters most at exactly the moment caching is supposed to
        help. When a popular prompt arrives 50 times at once with a cold cache,
        all 50 miss, all 50 hit the provider, and 49 of those responses are
        thrown away. Collapsing them into one upstream call turns a
        cache-stampede into a single request.
        """
        started = time.perf_counter()

        if self.cache is not None and not self.cache.is_cacheable(request):
            # Distinguishing "skipped" from "miss" matters: reporting a skip as
            # a miss makes the hit rate on a dashboard look like a cache that is
            # working badly, rather than one that was never consulted.
            metrics.cache_operations_total.labels(result="skipped").inc()
        elif self.cache is not None:
            hit = self.cache.lookup(request)
            if hit is not None:
                metrics.cache_operations_total.labels(result="hit").inc()
                metrics.requests_total.labels(outcome="cached").inc()
                metrics.request_duration_seconds.labels(outcome="cached").observe(
                    time.perf_counter() - started
                )
                metrics.tokens_saved_total.inc(hit.usage.total_tokens)
                return hit
            metrics.cache_operations_total.labels(result="miss").inc()

        key = self._exact_key(request)
        existing = self._inflight.get(key)
        if existing is not None:
            # Someone else is already fetching this exact request. Wait on
            # their result instead of issuing a duplicate.
            log.info("singleflight.joined")
            return await asyncio.shield(existing)

        future: asyncio.Future[ChatResponse] = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            response = await self.executor.complete(request)
        except BaseException as exc:
            # Waiters must see the failure too, otherwise they hang forever.
            if not future.done():
                future.set_exception(exc)
            metrics.requests_total.labels(outcome="error").inc()
            metrics.request_duration_seconds.labels(outcome="error").observe(
                time.perf_counter() - started
            )
            raise
        else:
            if not future.done():
                future.set_result(response)
            if self.cache is not None:
                self.cache.store(request, response)
            metrics.requests_total.labels(outcome="success").inc()
            metrics.request_duration_seconds.labels(outcome="success").observe(
                time.perf_counter() - started
            )
            metrics.tokens_total.labels(
                provider=response.provider, direction="prompt"
            ).inc(response.usage.prompt_tokens)
            metrics.tokens_total.labels(
                provider=response.provider, direction="completion"
            ).inc(response.usage.completion_tokens)
            return response
        finally:
            # Always release the slot. Leaving a completed future in the map
            # would turn single-flight into a permanent, unbounded cache that
            # never expires and never evicts.
            self._inflight.pop(key, None)
            # A future nobody awaited still needs its exception retrieved, or
            # asyncio logs "exception was never retrieved" on GC.
            if future.done() and not future.cancelled():
                future.exception()

    async def aclose(self) -> None:
        await self.executor.aclose()
