# LLM Gateway

**A reliability-focused proxy for LLM providers: hedged requests, circuit breaking, retry budgets, and semantic caching.**

Runs with **zero API keys** — a built-in mock provider with configurable failure
rates makes every reliability claim below reproducible on your own machine.

```bash
pip install -e ".[dev]" && pytest && uvicorn gateway.main:app
```

---

## Why this exists

Most LLM proxies focus on routing and cost tracking. This one focuses on the
harder problem: **staying up when providers don't.**

| Mechanism | Fixes | Why the others don't help |
|---|---|---|
| Retry | A request that failed | — |
| Failover | A provider that failed | Retrying a dead provider just fails again |
| **Hedging** | A request that is merely **slow** | Nothing has failed, so there's nothing to retry |
| Retry budget | Retries becoming the outage | Per-request retry limits can't see system-wide load |
| Circuit breaker | Paying a full timeout per doomed request | — |

Hedging is the centrepiece. Tail latency usually comes from one unlucky
*instance*, not a slow *service* — so the fix is to stop waiting and ask
someone else, while keeping the original in flight in case it lands first.
Fire the hedge at p95 and you duplicate only ~5% of requests.

## Status

Working today:

- ✅ Async FastAPI service, OpenAI-compatible API (point any client at it)
- ✅ SSE streaming with **commit-aware failover** — see below
- ✅ Circuit breaker with single-probe HALF_OPEN recovery
- ✅ Sliding-window retry budget (retry-storm protection)
- ✅ Hedged requests with automatic cancellation of the loser
- ✅ Exponential backoff with full jitter
- ✅ Configurable mock provider for chaos testing
- ✅ `mypy --strict` clean · `ruff` clean · 15 tests incl. property-based

In progress: semantic caching, cost ledger, Prometheus metrics, chaos benchmark
harness, Groq/Gemini adapters.

## The subtle part: streaming can't fail over

A normal request is atomic — if it fails, retry silently and the client never
knows. **A stream is not.** Once the first chunk ships, the client has seen part
of an answer; failing over now would splice two different answers together.

So the executor tracks a commit point. Before the first chunk, failover is safe.
After it, errors propagate. There's also no HTTP status left to use — the `200`
went out with the first byte — so mid-stream errors are delivered as an SSE
error event.

## Design decisions

- **SQLite by default, Postgres optional.** A portfolio project that needs
  Docker running before it demos is a project that doesn't get demoed.
- **Mock provider as a first-class component.** You cannot ask a real provider
  to fail 30% of requests, so you cannot demonstrate a breaker against one.
- **Permanent errors never retried, never counted against the breaker.** A
  malformed request is the client's fault; letting it trip the breaker would
  let one bad client take a healthy provider offline for everyone.
- **Retry budget over per-request retry counts.** See `docs/DEFENDING.md`.

## Docs

- [`docs/DEFENDING.md`](docs/DEFENDING.md) — the reasoning behind every design
  decision, plus an asyncio primer.

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

## License

MIT
