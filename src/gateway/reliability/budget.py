"""Retry budget.

Retries are the most common way a partial outage becomes a total one. If a
provider starts failing and every client retries twice, upstream load triples
at exactly the moment the provider is least able to absorb it. The retries
become the outage.

A budget makes retries a scarce resource: you may only retry in proportion to
how much traffic is *succeeding*. When things are healthy, retries are
effectively free. When everything is failing, retries stop almost entirely --
which is precisely the correct behaviour, and the opposite of what naive
per-request retry logic does.

Implemented as a sliding window over recent outcomes rather than a token
bucket, so the budget adapts to actual traffic volume without needing to be
tuned to a requests-per-second figure that changes constantly.
"""

from __future__ import annotations

import time
from collections import deque


class RetryBudget:
    def __init__(
        self, *, ratio: float = 0.15, window_seconds: float = 10.0, min_allowance: int = 3
    ) -> None:
        self.ratio = ratio
        self.window_seconds = window_seconds
        self.min_allowance = min_allowance
        """A small floor so a cold or very low-traffic gateway can still retry."""

        self._successes: deque[float] = deque()
        self._retries: deque[float] = deque()

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for q in (self._successes, self._retries):
            while q and q[0] < cutoff:
                q.popleft()

    def record_success(self) -> None:
        now = time.monotonic()
        self._evict(now)
        self._successes.append(now)

    def try_consume(self) -> bool:
        """Attempt to spend one retry. Returns False if the budget is exhausted."""
        now = time.monotonic()
        self._evict(now)

        allowance = max(self.min_allowance, int(len(self._successes) * self.ratio))
        if len(self._retries) >= allowance:
            return False

        self._retries.append(now)
        return True

    @property
    def stats(self) -> dict[str, int]:
        now = time.monotonic()
        self._evict(now)
        return {
            "successes_in_window": len(self._successes),
            "retries_in_window": len(self._retries),
            "allowance": max(self.min_allowance, int(len(self._successes) * self.ratio)),
        }
