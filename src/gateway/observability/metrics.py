"""Prometheus metrics.

Metric design is mostly about restraint. Every label combination creates a new
time series, and a label with unbounded values -- a prompt, a user id, a model
name accepted from client input -- will grow series until the scrape times out
and the monitoring falls over. That failure mode is called cardinality
explosion, and it takes down monitoring exactly when you need it.

So every label here has a small, bounded set of possible values: provider names
come from configuration, outcomes from a fixed enum.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# A private registry rather than the global default. The default is process-wide
# mutable state, so importing this module twice under different names -- or
# running two apps in one process during tests -- raises "Duplicated timeseries".
REGISTRY = CollectorRegistry()

requests_total = Counter(
    "gateway_requests_total",
    "Requests handled, by outcome.",
    ["outcome"],  # success | error | cached
    registry=REGISTRY,
)

provider_requests_total = Counter(
    "gateway_provider_requests_total",
    "Requests sent upstream, by provider and outcome.",
    ["provider", "outcome"],  # success | transient | permanent | rate_limited
    registry=REGISTRY,
)

request_duration_seconds = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end request latency.",
    ["outcome"],
    # Explicit buckets, tuned for LLM latencies. The library default tops out
    # at 10s, which puts every slow LLM call into +Inf and makes p99 unusable.
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)

hedges_total = Counter(
    "gateway_hedges_total",
    "Hedged requests fired and won.",
    ["result"],  # fired | won
    registry=REGISTRY,
)

circuit_state = Gauge(
    "gateway_circuit_state",
    "Circuit breaker state per provider (0=closed, 1=half_open, 2=open).",
    ["provider"],
    registry=REGISTRY,
)

cache_operations_total = Counter(
    "gateway_cache_operations_total",
    "Cache lookups by result.",
    ["result"],  # hit | miss | skipped
    registry=REGISTRY,
)

tokens_total = Counter(
    "gateway_tokens_total",
    "Tokens consumed upstream, by provider and direction.",
    ["provider", "direction"],  # prompt | completion
    registry=REGISTRY,
)

tokens_saved_total = Counter(
    "gateway_tokens_saved_total",
    "Tokens not spent upstream because the cache served the response.",
    registry=REGISTRY,
)

_STATE_VALUES = {"closed": 0, "half_open": 1, "open": 2}

# Last value pushed to the hedge counters, so scrape-time sampling can be
# converted into the increments a Counter requires.
_hedges_reported = {"fired": 0, "won": 0}


def record_hedges(fired: int, won: int) -> None:
    """Sync the hedge counters from the executor's cumulative totals.

    The executor owns plain integers and does not import this module -- keeping
    the reliability layer free of observability dependencies makes it far
    easier to test. Converting those cumulative values into Counter increments
    is this function's job.

    Note the delta arithmetic: a Counter may only ever be incremented. Setting
    its internal value directly (`._value.set(...)`) reaches into private API
    and would silently break `rate()` if the value ever moved backwards.
    """
    for label, current in (("fired", fired), ("won", won)):
        delta = current - _hedges_reported[label]
        if delta > 0:
            hedges_total.labels(result=label).inc(delta)
            _hedges_reported[label] = current


def record_circuit_states(states: dict[str, str]) -> None:
    """Publish breaker states as a gauge.

    Sampled at scrape time rather than pushed on every transition: a gauge only
    needs its current value, and updating it on each state change would couple
    the reliability layer to the metrics layer for no benefit.
    """
    for provider, state in states.items():
        circuit_state.labels(provider=provider).set(_STATE_VALUES.get(state, 0))
