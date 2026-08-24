"""Semantic cache evaluation.

A cache hit rate on its own is a meaningless number. Set the threshold to 0 and
you get 100% hits and a completely broken product, because every request
returns somebody else's answer. Hit rate is only interpretable alongside
**precision**: of the hits served, how many were actually correct?

So this measures both, sweeping the similarity threshold to expose the whole
tradeoff curve, and picks an operating point deliberately rather than by
guessing a round number.

    python -m chaos.cache_bench

Note the asymmetry that drives the choice: a false hit returns a WRONG ANSWER
to a user, while a false miss just costs one API call. Those are not equally
bad, so the operating point is chosen for high precision, not best F1.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from gateway.cache.embedder import Embedder, HashingEmbedder, cosine, default_embedder

RESULTS_DIR = Path(__file__).parent / "results"

# --- Labelled evaluation set ----------------------------------------------
# (prompt_a, prompt_b, should_be_treated_as_the_same_question)
#
# The negatives are deliberately HARD: high word overlap, different meaning.
# Easy negatives ("what is python" vs "how do I bake bread") would let any
# threshold look perfect and tell us nothing.

PAIRS: list[tuple[str, str, bool]] = [
    # --- Positives: same question, different wording ---
    ("what is a circuit breaker", "what's a circuit breaker?", True),
    ("how do I reset my password", "how can I reset my password", True),
    ("what is a retry budget", "what is a retry budget?", True),
    ("explain hedged requests", "explain hedged requests.", True),
    ("list the top 5 python web frameworks", "list the top five python web frameworks", True),
    ("how do i configure the hedge delay", "How do I configure the hedge delay?", True),
    ("what does p95 latency mean", "what does p95 latency mean???", True),
    ("summarise this quarter's revenue", "summarize this quarter's revenue", True),
    ("is the gateway open source", "is the gateway open-source", True),
    ("show me the error rate", "show me the error rate please", True),
    # --- Hard negatives: near-identical wording, different meaning ---
    ("is redis better than postgres", "is postgres better than redis", False),
    ("how do I enable caching", "how do I disable caching", False),
    ("convert celsius to fahrenheit", "convert fahrenheit to celsius", False),
    ("what is the capital of Australia", "what is the capital of Austria", False),
    ("increase the retry budget", "decrease the retry budget", False),
    ("send the report to alice", "send the report to bob", False),
    ("what is the p50 latency", "what is the p99 latency", False),
    ("how do I open a circuit", "how do I close a circuit", False),
    ("translate this to french", "translate this to german", False),
    ("delete the cache entry", "create the cache entry", False),
    # --- Identifier negatives: same template, different entity ---
    # These were added AFTER the traffic replay exposed the failure. The
    # sensitivity sweep showed the hit rate barely moving as the prompt
    # universe grew from 200 to 6000 distinct prompts, which is impossible for
    # genuinely diverse traffic -- the cache was serving one tenant's answer to
    # another. Templated prompts that differ only by an ID are the single most
    # dangerous input for a semantic cache, and the labelled set did not
    # originally contain any.
    ("restart service 41", "restart service 87", False),
    ("what is the status of order 4471", "what is the status of order 8823", False),
    ("show metrics for tenant alpha", "show metrics for tenant beta", False),
    ("how do I tune latency for service 12", "how do I tune latency for service 990", False),
    ("summarise invoice 1023", "summarise invoice 7781", False),
    # --- Easy negatives: unrelated ---
    ("what is a circuit breaker", "what is the capital of France", False),
    ("how do I reset my password", "write me a poem about the sea", False),
    ("explain hedged requests", "what time is the meeting tomorrow", False),
    ("show me the error rate", "recommend a good italian restaurant", False),
]


@dataclass(slots=True)
class ThresholdResult:
    threshold: float
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        """Of the cache hits served, how many were correct?

        This is the number that matters. Low precision means users getting
        answers to questions they did not ask.
        """
        served = self.true_positives + self.false_positives
        return self.true_positives / served if served else 1.0

    @property
    def recall(self) -> float:
        """Of the hits we could have served, how many did we catch?"""
        available = self.true_positives + self.false_negatives
        return self.true_positives / available if available else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def sweep(embedder: Embedder) -> list[ThresholdResult]:
    """Evaluate the CACHE, not the embedder in isolation.

    An earlier version scored raw cosine similarity alone, which measured only
    one component and made the identifier failures look unfixable by any
    threshold. The cache also refuses to compare prompts whose entity
    signatures differ, so the evaluation has to model that guard too --
    otherwise it is calibrating a system that does not exist.
    """
    from gateway.cache.semantic import SemanticCache

    scored = [
        (
            cosine(embedder.embed(a), embedder.embed(b)),
            SemanticCache._entity_signature(a) == SemanticCache._entity_signature(b),  # noqa: SLF001
            same,
        )
        for a, b, same in PAIRS
    ]

    results = []
    for step in range(50, 100):
        threshold = step / 100
        r = ThresholdResult(threshold=threshold)
        for score, same_entities, same in scored:
            served = same_entities and score >= threshold
            if served and same:
                r.true_positives += 1
            elif served and not same:
                r.false_positives += 1
            elif not served and same:
                r.false_negatives += 1
            else:
                r.true_negatives += 1
        results.append(r)
    return results


def choose_operating_point(results: list[ThresholdResult]) -> ThresholdResult:
    """Lowest threshold that still achieves perfect precision.

    Not best-F1. F1 treats a false hit and a false miss as equally bad, which
    is wrong here: one is a correctness bug, the other is one extra API call.
    We take perfect precision first, then the most recall available under that
    constraint.
    """
    perfect = [r for r in results if r.precision >= 1.0 and r.recall > 0]
    if not perfect:
        return max(results, key=lambda r: r.f1)
    return max(perfect, key=lambda r: r.recall)


# --- Traffic replay --------------------------------------------------------


def _synthetic_corpus(rng: random.Random, size: int) -> list[str]:
    """A long tail of distinct prompts.

    Real traffic is mostly unique. An evaluation whose prompt universe is only
    a few dozen strings will report a near-perfect hit rate no matter how bad
    the cache is, because every request is a repeat by construction.
    """
    subjects = ["latency", "billing", "the retry budget", "streaming", "auth", "quotas",
                "the circuit breaker", "webhooks", "rate limits", "the cache"]
    verbs = ["configure", "debug", "monitor", "disable", "tune", "explain", "audit"]
    return [
        f"how do I {rng.choice(verbs)} {rng.choice(subjects)} for service {i}"
        for i in range(size)
    ]


def _paraphrase(text: str, rng: random.Random) -> str:
    """Surface variation of the kind a cache should see through."""
    choice = rng.random()
    if choice < 0.3:
        return text + "?"
    if choice < 0.5:
        return text.capitalize()
    if choice < 0.7:
        return text + " please"
    return text.replace("how do I", "how can I")


def replay(
    embedder: Embedder, threshold: float, *, n: int = 1500, corpus_size: int = 600
) -> dict[str, float | int | str]:
    """Estimate hit rate on traffic with a stated popularity distribution.

    Prompt popularity follows Zipf(s=1) over `corpus_size` distinct prompts: the
    k-th most popular prompt is requested proportionally to 1/k. Repeats arrive
    as paraphrases rather than byte-identical strings, so this exercises the
    cache rather than a dictionary lookup.

    The hit rate is only meaningful WITH that assumption attached, and it is
    reported alongside the result for exactly that reason. An earlier version
    sampled from a Pareto distribution that put ~60% of all traffic on a single
    prompt and duly reported a 98.7% hit rate -- a number that measured the
    sampler, not the cache.
    """
    from gateway.cache.semantic import SemanticCache
    from gateway.types import ChatRequest, ChatResponse, Message, Usage

    rng = random.Random(42)
    corpus = _synthetic_corpus(rng, corpus_size)
    # Zipf(s=1): weight of rank k is 1/k.
    weights = [1.0 / (k + 1) for k in range(corpus_size)]
    draws = rng.choices(corpus, weights=weights, k=n)

    # Capping entries also bounds lookup cost: the scan is O(entries), so an
    # unbounded cache would get progressively slower per request.
    cache = SemanticCache(embedder=embedder, threshold=threshold, max_entries=400)

    upstream_calls = 0
    for text in draws:
        if rng.random() < 0.6:
            text = _paraphrase(text, rng)

        request = ChatRequest(messages=(Message(role="user", content=text),), temperature=0.0)
        if cache.lookup(request) is not None:
            continue
        upstream_calls += 1
        cache.store(
            request,
            ChatResponse(
                content="answer",
                model="m",
                provider="p",
                usage=Usage(prompt_tokens=120, completion_tokens=400),
            ),
        )

    return {
        "requests": n,
        "distinct_prompts": corpus_size,
        "popularity_model": "zipf(s=1)",
        "upstream_calls": upstream_calls,
        "hit_rate": round(cache.stats.hit_rate, 4),
        "tokens_saved": cache.stats.tokens_saved,
        "cost_reduction": round(1 - upstream_calls / n, 4),
    }


# --- Chart -----------------------------------------------------------------


def curve_svg(results: list[ThresholdResult], chosen: ThresholdResult) -> str:
    """Precision and recall against threshold."""
    w, h, pad = 700, 300, 50
    plot_w, plot_h = w - pad * 2, h - pad * 2
    lo, hi = results[0].threshold, results[-1].threshold

    def x(t: float) -> float:
        return pad + (t - lo) / (hi - lo) * plot_w

    def y(v: float) -> float:
        return pad + (1 - v) * plot_h

    def path(values: list[tuple[float, float]]) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'}{x(t):.1f},{y(v):.1f}" for i, (t, v) in enumerate(values)
        )

    precision = [(r.threshold, r.precision) for r in results]
    recall = [(r.threshold, r.recall) for r in results]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<line x1="{pad}" y1="{y(0)}" x2="{pad + plot_w}" y2="{y(0)}" stroke="#cbd5e1"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{y(0)}" stroke="#cbd5e1"/>',
    ]
    for v in (0.0, 0.5, 1.0):
        parts.append(
            f'<text x="{pad - 8}" y="{y(v) + 4}" font-size="11" text-anchor="end" '
            f'fill="#94a3b8">{v:.1f}</text>'
        )
    for t in (lo, (lo + hi) / 2, hi):
        parts.append(
            f'<text x="{x(t)}" y="{y(0) + 18}" font-size="11" text-anchor="middle" '
            f'fill="#94a3b8">{t:.2f}</text>'
        )

    parts.append(
        f'<line x1="{x(chosen.threshold)}" y1="{pad}" x2="{x(chosen.threshold)}" '
        f'y2="{y(0)}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 3"/>'
        f'<text x="{x(chosen.threshold) + 6}" y="{pad + 12}" font-size="11" '
        f'fill="#f59e0b" font-weight="600">chosen {chosen.threshold:.2f}</text>'
    )
    parts.append(f'<path d="{path(precision)}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>')
    parts.append(f'<path d="{path(recall)}" fill="none" stroke="#94a3b8" stroke-width="2.5"/>')
    parts.append(
        f'<text x="{pad}" y="{h - 12}" font-size="12" fill="#3b82f6" font-weight="600">'
        f'— precision</text>'
        f'<text x="{pad + 90}" y="{h - 12}" font-size="12" fill="#94a3b8" font-weight="600">'
        f'— recall</text>'
        f'<text x="{pad}" y="{pad - 22}" font-size="13" font-weight="700" fill="#334155">'
        f'Cache threshold calibration</text>'
        "</svg>"
    )
    return "".join(parts)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    embedder = default_embedder()
    fallback = isinstance(embedder, HashingEmbedder)

    print("\n\033[1mSemantic cache calibration\033[0m")
    print(f"  embedder: {embedder.name}")
    if fallback:
        print("  (install sentence-transformers for true semantic matching)")

    results = sweep(embedder)
    chosen = choose_operating_point(results)

    print(f"\n  {'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8}")
    for r in results:
        if round(r.threshold * 100) % 5 == 0:
            mark = "  <- chosen" if r.threshold == chosen.threshold else ""
            print(
                f"  {r.threshold:>10.2f} {r.precision:>10.2f} "
                f"{r.recall:>8.2f} {r.f1:>8.2f}{mark}"
            )

    print(
        f"\n  Operating point: {chosen.threshold:.2f} "
        f"(precision {chosen.precision:.0%}, recall {chosen.recall:.0%})"
    )
    print("  Chosen for perfect precision, not best F1 - a false hit is a wrong answer.")

    # Hit rate is a property of TRAFFIC SHAPE, not of the cache alone. Quoting
    # a single figure invites a number that flatters the implementation, so
    # sweep the size of the prompt universe and show the sensitivity.
    print("\n  Traffic replay - hit rate vs. traffic diversity (Zipf s=1, 1500 requests)")
    print(f"    {'distinct prompts':>18} {'upstream':>10} {'hit rate':>10} {'cost cut':>10}")
    replays: list[dict[str, float | int | str]] = []
    for corpus_size in (200, 600, 2000, 6000):
        stats = replay(embedder, chosen.threshold, corpus_size=corpus_size)
        replays.append(stats)
        print(
            f"    {stats['distinct_prompts']:>18} {stats['upstream_calls']:>10} "
            f"{stats['hit_rate']:>9.1%} {stats['cost_reduction']:>9.1%}"
        )
    print("\n  The cache is worth most on repetitive traffic and least on diverse")
    print("  traffic. Any single headline hit rate hides that.")

    payload = {
        "embedder": embedder.name,
        "using_fallback_embedder": fallback,
        "operating_point": {
            "threshold": chosen.threshold,
            "precision": round(chosen.precision, 4),
            "recall": round(chosen.recall, 4),
            "f1": round(chosen.f1, 4),
        },
        "sweep": [
            {
                "threshold": r.threshold,
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1": round(r.f1, 4),
            }
            for r in results
        ],
        "replay": replays,
        "labelled_pairs": len(PAIRS),
    }
    (RESULTS_DIR / "cache.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / "cache.svg").write_text(curve_svg(results, chosen), encoding="utf-8")

    print(f"\n  wrote {RESULTS_DIR / 'cache.json'}")
    print(f"  wrote {RESULTS_DIR / 'cache.svg'}\n")


if __name__ == "__main__":
    main()
