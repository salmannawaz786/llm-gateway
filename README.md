# LLM Gateway

**A reliability-focused proxy for LLM providers: hedged requests, circuit breaking, and retry budgets.**

Against a provider failing **30% of requests**, the gateway sustains a **100% success rate**
while cutting p95 latency from **236ms → 64ms**. During a total outage it inflicts
**18× less load** on the failing provider than a conventional retry loop.

Every number below is reproducible on your machine in about 30 seconds, with **no API keys**:

```bash
pip install -e ".[dev]"
python -m chaos.run
```

![chaos benchmark results](chaos/results/chaos.svg)

---

## Measured results

Each row is an A/B against an identical provider with an identical random seed, so the
only variable is the reliability layer. `upstream` counts requests that actually reached
a provider — the cost axis.

**Degraded provider — 30% error rate, 400 requests**

| | success | p50 | p95 | upstream calls |
|---|---|---|---|---|
| Naive retry loop | 96.8% | 63ms | 236ms | 587 |
| **Gateway** | **100.0%** | **61ms** | **64ms** | 549 |

**Tail latency — 10% of requests take 2s, 400 requests**

| | p50 | p95 | p99 | upstream calls |
|---|---|---|---|---|
| Hedging off | 61ms | 2004ms | 2015ms | 400 |
| **Hedging on** | 61ms | **269ms** | **278ms** | 441 (+10%) |

> 41 hedges fired, 41 won. A **7.4× p95 improvement for 10% extra upstream calls** —
> because the hedge only fires on requests already proven slow.

**Total outage — provider 100% down, 300 requests**

| | upstream calls |
|---|---|
| Naive retry loop | 900 |
| **Gateway** (retry budget + circuit breaker) | **50** |

> Nobody succeeds here; that's the point. The question isn't "who stays up" but
> "who makes the outage worse." A naive retry loop **triples** load on a provider that
> is already failing — that's a retry storm. The budget collapses as successes vanish,
> and the breaker stops paying a timeout per doomed request.

## Why this exists

Most LLM proxies focus on routing and cost tracking. This one focuses on the harder
problem: **staying up when providers don't.**

| Mechanism | Fixes | Why the others don't help |
|---|---|---|
| Retry | A request that failed | — |
| Failover | A provider that failed | Retrying a dead provider just fails again |
| **Hedging** | A request that is merely **slow** | Nothing has failed, so there's nothing to retry |
| Retry budget | Retries *becoming* the outage | Per-request retry limits can't see system-wide load |
| Circuit breaker | Paying a full timeout per doomed request | — |

Hedging is the centrepiece. Tail latency usually comes from one unlucky *instance*, not
a slow *service* — so the fix is to stop waiting and ask someone else, while keeping the
original in flight in case it lands first.

## The subtle part: streaming can't fail over

A normal request is atomic — if it fails, retry silently and the client never knows.
**A stream is not.** Once the first chunk ships, the client has seen part of an answer;
failing over now would splice two different answers together.

So the executor tracks a commit point. Before the first chunk, failover is safe. After
it, errors propagate. There's also no HTTP status left to use — the `200` went out with
the first byte — so mid-stream errors are delivered as an SSE error event.

## Running it

```bash
pip install -e ".[dev]"

pytest                          # 15 tests, incl. property-based
mypy && ruff check .            # strict, clean
python -m chaos.run             # regenerate the benchmarks above
uvicorn gateway.main:app        # OpenAI-compatible API on :8000
```

The gateway runs on built-in mock providers by default, so it needs **no API keys and no
Docker**. Groq and Gemini adapters (both free tiers) activate when their keys are set.

```bash
curl localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

`GET /healthz` exposes live breaker states, retry-budget headroom, and hedge counters.

## Design decisions

- **The mock provider is a first-class component, not a test fixture.** You cannot ask a
  real provider to fail 30% of requests on command, so you cannot demonstrate a circuit
  breaker against one. Every benchmark above exists because the mock does.
- **SQLite by default, Postgres optional.** A portfolio project that needs Docker running
  before it demos is a project that doesn't get demoed.
- **Permanent errors are never retried and never counted against the breaker.** A
  malformed request is the client's fault; letting it trip the breaker would let one bad
  client take a healthy provider offline for everyone.
- **Retry budgets over per-request retry counts.** Per-request limits are fine when one
  request fails and catastrophic when everything does.
- **Losing hedges are cancelled in a `finally` block.** Otherwise every hedged request
  keeps burning upstream tokens for a response nobody will read.

## Docs

- [`docs/DEFENDING.md`](docs/DEFENDING.md) — the reasoning behind every design decision,
  plus an asyncio primer.

## Testing

The tests target failure modes, not happy paths:

```
test_breaker_half_open_admits_exactly_one_probe   10 concurrent callers, exactly 1 probe
test_budget_blocks_retry_storm                    retries collapse during an outage
test_budget_never_exceeds_allowance               property-based (Hypothesis)
test_hedge_fires_only_when_primary_is_slow        hedging triggers on latency
test_no_hedge_when_primary_is_fast                and NOT on healthy traffic
test_losing_hedge_is_cancelled                    no leaked tasks, no wasted tokens
test_stream_fails_over_before_first_chunk         commit-aware failover
test_handles_concurrent_load                      100 requests, nothing serialised
```

## Status

Working: async FastAPI service, OpenAI-compatible API, SSE streaming with commit-aware
failover, circuit breaker with single-probe recovery, sliding-window retry budget, hedged
requests with cancellation, full-jitter backoff, configurable mock provider, chaos
benchmark harness. `mypy --strict` and `ruff` clean.

In progress: semantic caching, cost ledger, Prometheus metrics, Groq/Gemini adapters.

## License

MIT
