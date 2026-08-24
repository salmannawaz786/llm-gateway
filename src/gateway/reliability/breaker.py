"""Circuit breaker.

The problem it solves: when a provider is down, every request still pays the
full timeout before failing. At any real concurrency that means thousands of
tasks parked on a dead socket, holding connections and memory, all to discover
something the first ten requests already proved.

A breaker makes failure cheap. Once a provider looks dead, calls fail
instantly, freeing the gateway to route elsewhere.

States:

    CLOSED  --failures exceed threshold-->  OPEN
    OPEN    --after recovery timeout----->  HALF_OPEN
    HALF_OPEN --trial succeeds----------->  CLOSED
    HALF_OPEN --trial fails-------------->  OPEN

HALF_OPEN exists so recovery costs exactly one probe request. Going straight
from OPEN to CLOSED would slam a just-recovering provider with the full backlog
and knock it over again -- the classic thundering herd.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum

from gateway.types import CircuitOpenError


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_seconds: float = 15.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds

        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        # Guards the state transitions below. Without it, two tasks can both
        # observe HALF_OPEN and both send a "single" trial request.
        self._lock = asyncio.Lock()

    @property
    def state(self) -> BreakerState:
        return self._state

    async def allows_request(self) -> bool:
        """Whether a call may proceed, transitioning OPEN -> HALF_OPEN if due."""
        async with self._lock:
            if self._state is BreakerState.CLOSED:
                return True

            if self._state is BreakerState.OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_seconds:
                    self._state = BreakerState.HALF_OPEN
                    return True  # this caller is the single trial request
                return False

            # HALF_OPEN: a trial is already in flight, so hold everyone else
            # back until it resolves.
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self._consecutive_failures = 0
            self._state = BreakerState.CLOSED

    async def record_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if (
                self._state is BreakerState.HALF_OPEN
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._state = BreakerState.OPEN
                self._opened_at = time.monotonic()

    async def guard(self) -> None:
        """Raise if the circuit is open. Convenience for call sites."""
        if not await self.allows_request():
            raise CircuitOpenError(f"circuit open for provider {self.name!r}")
