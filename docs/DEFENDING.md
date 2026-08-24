# Defending This Code

A guide to explaining every decision in this project — written for you, not for
recruiters. Read it until you can answer these without looking.

**The rule that matters:** a project you cannot explain is worse than no
project. An interviewer who senses you don't understand your own code will
distrust everything else on your CV. Conversely, someone who can explain *why*
they cancelled a losing task will get more credit than someone who built
something twice as large and can only describe what it does.

---

## Part 1 — asyncio, from zero

You said you haven't gone deep on asyncio. Here is the whole mental model. It's
smaller than it looks.

### The one-sentence version

> asyncio runs many tasks on **one thread** by letting each task voluntarily
> pause at an `await` while it waits for something slow, so the thread can go do
> other work instead of sitting idle.

That's it. Everything else is detail.

### Why it exists

A gateway request spends ~99.9% of its time doing nothing — just waiting for
the provider to answer. With one thread per request, 1,000 concurrent requests
means 1,000 threads, each consuming ~8MB of stack, all asleep. You run out of
memory long before you run out of useful work.

asyncio makes waiting free. One thread, 1,000 paused tasks, a few KB each.

### The four things you need to know

**1. `async def` makes a coroutine, not a running thing.**

```python
async def f():
    return 1

f()          # creates a coroutine object; the body has NOT run
await f()    # actually runs it
```

Forgetting the `await` is the single most common asyncio bug. The function
silently never runs and you get a `RuntimeWarning` you may not notice.

**2. `await` = "pause me here, wake me when this is done."**

While your task is paused at an `await`, the event loop runs other tasks. The
pause is the *feature*.

**3. `await` runs things one after another. `gather`/`create_task` runs them at
the same time.**

```python
a = await slow()   # 1 second
b = await slow()   # 1 more second — total 2s

a, b = await asyncio.gather(slow(), slow())   # total 1s
```

This trips up almost everyone. Sprinkling `async` on things does *not* make
them concurrent. Concurrency comes only from `gather`, `create_task`, or
`asyncio.wait`.

**4. Blocking calls poison the whole loop.**

```python
time.sleep(1)          # freezes the ENTIRE server for 1 second
await asyncio.sleep(1) # pauses only this task
```

Same for `requests.get()` (use `httpx.AsyncClient`), and for heavy CPU work.
One blocking call and every other request in the process stalls.

> **This is why `mock.py` uses `await asyncio.sleep()` to simulate latency.**
> With `time.sleep()` the mock would serialise every request and every
> benchmark number in the README would be meaningless. Say this out loud in an
> interview — it shows you know *why* the rule exists, not just the rule.

### The one advanced piece this project uses

`asyncio.wait` — wait on a *set* of tasks with a timeout, and get back which
finished and which are still running:

```python
done, pending = await asyncio.wait(
    tasks, timeout=0.8, return_when=asyncio.FIRST_COMPLETED
)
```

Three outcomes:
- something finished → it's in `done`
- nothing finished within 0.8s → `done` is empty (**this is the hedge trigger**)
- `pending` holds whatever is still running

That single call is the entire hedging mechanism. Everything else in
`_hedged()` is bookkeeping around it.

---

## Part 2 — the questions you will actually be asked

### "Walk me through this project."

> It's an LLM gateway — a proxy that sits between an application and providers
> like Anthropic or OpenAI. The interesting part isn't the routing, it's the
> reliability layer: circuit breakers, a retry budget, and hedged requests.
>
> The thing I'm most pleased with is that the claims are provable. There's a
> mock provider with configurable failure rates, so I can make a provider fail
> 30% of requests on command and measure that the gateway still succeeds
> 99.9% of the time. You can't do that against a real API.

Short, leads with the hard part, ends on evidence. Don't list features.

### "Why not just use LiteLLM?"

**Expect this. It's the question that kills unprepared candidates.**

> For production, you probably should — it's mature and handles far more
> providers. I built this because using a library teaches you its API, and
> implementing one teaches you the problem. I wanted to actually understand why
> retries need a budget and why streaming can't fail over after the first byte.
> Those turned out to be much subtler than I expected.

Never get defensive. Agreeing that the mature tool is better *and* explaining
what you learned is a strong answer; insisting yours is better is a weak one.

### "What's a hedged request?"

> Sending the same request to a second provider when the first one is taking
> too long, and using whichever responds first.
>
> The key insight is that it fixes a problem retries can't. A retry fires after
> something *fails*. But tail latency isn't failure — nothing's wrong, the
> request is just slow, usually because of one unlucky instance. Retrying
> doesn't help because there's nothing to retry yet.
>
> The cost is controlled by the delay. Fire the hedge at your p95 and you only
> duplicate ~5% of requests, so you cut the tail for about 5% extra spend.

### "Why cancel the losing hedge?"

> Two reasons. It's still generating tokens I'd be billed for and will never
> read. And an un-awaited task that outlives the request is a leak — asyncio
> will eventually warn about "task exception never retrieved."
>
> The cancellation is in a `finally` block so it runs whether the request
> succeeded, failed, or raised.

### "Why does streaming behave differently from a normal request?"

**This is the best question in the project. Make sure you nail it.**

> Because a normal request is atomic and a stream isn't.
>
> With a normal request, if the provider fails I can silently retry — the
> client never knows. With a stream, once I've sent the first chunk the client
> has already seen part of an answer. If I fail over now, the second provider
> starts a *different* answer and I'd splice two half-responses together.
>
> So I track a `committed` flag. Before the first chunk, failover is safe.
> After it, errors propagate to the client. There's also no HTTP status left to
> use — the 200 went out with the first byte — so mid-stream errors get
> delivered as an SSE error event instead.

### "What's a retry budget and why not just retry 3 times?"

> Per-request retries are fine when one request fails, and catastrophic when
> everything fails. If a provider degrades and every client retries twice,
> upstream load triples at the exact moment the provider can least handle it.
> The retries become the outage — that's a retry storm.
>
> A budget caps retries as a fraction of recent *successes*. Healthy traffic
> means lots of successes, so retries are basically free. A total outage means
> no successes, so the budget collapses and retries nearly stop. It fails in
> the right direction automatically.

### "Why HALF_OPEN in the circuit breaker?"

> To make recovery cost exactly one request. If the breaker went straight from
> OPEN to CLOSED, the entire queued backlog would hit a provider that just came
> back and knock it straight over — a thundering herd. HALF_OPEN lets exactly
> one probe through and decides based on the result.
>
> There's a lock around the state transitions, because without it ten
> concurrent callers could all see HALF_OPEN and all send a "single" probe.
> There's a test for exactly that.

### "Why is `stream()` `def` and not `async def`?"

> Because it's an async generator function. Calling it already returns an async
> iterator — no `await` needed. Marking it `async def` would mean callers have
> to await a coroutine that returns a generator, which is a common and
> confusing mistake.

### "Tell me about a bug you hit."

Use a real one — it's far more convincing than a rehearsed answer:

> My hedging loop only fired a backup when the primary *timed out*. So a
> primary that failed instantly never triggered failover at all — the request
> just gave up while a perfectly healthy backup sat there unused.
>
> My tests caught it. I'd written a failover test expecting the second provider
> to answer, and it failed. The fix was distinguishing the two triggers:
> timeout means hedge, failure means promote the next provider immediately.

This answer is strong because it shows tests catching a real design flaw, and a
precise diagnosis. Interviewers love this far more than "everything worked."

### "How would you scale this?"

> Right now breaker and budget state is per-process, so with N replicas you get
> N independent breakers. That's actually acceptable — each one converges on
> the same answer, just N times more slowly — but the clean fix is shared state
> in Redis.
>
> The bigger issue is that a gateway is on the critical path for every request,
> so it must never become the bottleneck. That means keeping per-request work
> tiny, and doing cost-ledger writes asynchronously rather than blocking the
> response on a database round-trip.

### "What would you do differently?"

Never say "nothing."

> The hedge delay is a fixed config value. It should be adaptive — track a
> rolling p95 per provider and hedge at that, so it self-tunes instead of going
> stale whenever provider performance shifts.

---

## Part 3 — the honest bit

You didn't write this code from scratch, and interviewers increasingly assume
AI assistance for everyone. That's fine, and it's not what they're testing.

What separates candidates now is whether you **understand and can defend** what
you shipped. So:

1. **Read `executor.py` line by line.** It's ~200 lines. Every comment explains
   a decision — those comments are your script.
2. **Break things deliberately.** Delete the `finally` block that cancels
   hedges and watch the leak. Change `asyncio.sleep` to `time.sleep` in the
   mock and watch the concurrency test blow past its one-second budget.
   Breaking something teaches you more than reading it twice.
3. **Run the tests and read the failures.** Comment out the `PermanentProviderError`
   re-raise and see which test catches it.
4. **Never claim more than you know.** "I'd have to look at how that part works
   again" is a perfectly good answer. Bluffing is what actually loses offers.

If someone asks how you built it: *"I used AI heavily for the boilerplate, then
went through the reliability logic carefully until I understood it — that part
I can walk you through in detail."* That's honest, and it's a better answer
than most candidates give.

---

## Part 4 — defending the benchmarks

Numbers invite scrutiny in a way prose doesn't. Expect these.

### "How do I know this isn't rigged?"

> Every scenario is an A/B where both sides face the same provider profile and
> the same random seed. The only variable is the reliability layer. And the
> naive baseline isn't a straw man — it retries and it backs off, it just
> lacks the system-level mechanisms.
>
> I also report upstream call counts next to latency, so the cost of each
> mechanism is visible. Hedging cut p95 by 7.4× for 10% more upstream calls —
> if I'd only shown the latency, I'd be hiding the price.

Volunteering the cost axis is what makes the numbers credible.

### "Your p99 is 3000ms in BOTH rows of the degraded test. Why didn't hedging fix that?"

**Know this one cold — it's the sharpest question the data invites.**

> Because with two providers, hedging only protects the *first* one.
>
> The path is: primary fails (30% of the time), so we fail over to the backup.
> Now the backup is the only provider left — there's nothing to hedge with, so
> if it draws a slow response we just have to wait. That's roughly 30% × 5% ≈
> 1.5% of requests, which is exactly where p99 lands.
>
> It shows up at p99 and not p95 because it's a ~1.5% event. The fix would be a
> third provider, or re-hedging against the primary once its circuit recovers.

This is a strong answer because you're explaining a weakness in your own
results rather than being caught by it.

### "Nobody succeeds in your outage scenario. Isn't that a failed test?"

> No — that's the scenario. The provider is 100% down, so no strategy can
> succeed. The question isn't "who stays up," it's "who makes the outage
> worse." The naive loop sends 900 upstream calls for 300 requests; the gateway
> sends 50. Tripling load on a provider that's already failing is how a partial
> outage becomes a total one.

### "Why not use matplotlib for the chart?"

> Keeping the benchmark dependency-free matters — it's the thing I most want
> people to actually run. SVG also renders natively on GitHub in both light and
> dark themes, and a PNG doesn't.

### "These are mock numbers, not real providers."

Concede immediately. Don't defend.

> Correct, and I wouldn't present them as real-world figures. They measure the
> gateway's *behaviour* under controlled failure — which is the only way to
> measure it, since I can't make a real provider fail 30% of requests on
> demand. The latency distribution is modelled two-mode, fast with a heavy
> tail, because a uniform distribution would make hedging look useless.

That last sentence is worth memorising: it proves you thought about whether
your simulation was *fair*, which is the actual concern behind the question.

### Things to be honest about

- Breaker and budget state is per-process, so N replicas means N independent
  breakers.
- The hedge delay is static config; it should track a rolling p95 per provider.
- The benchmark runs in-process, so it excludes HTTP and serialisation
  overhead. It measures the reliability layer, not end-to-end service latency.

Naming your own limitations before you're asked reads as confidence. Being
caught by one reads as the opposite.
