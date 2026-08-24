"""Tests for the semantic cache and single-flight layer.

The bias throughout: a false hit is a correctness bug, a miss is only a cost.
So the tests that matter most are the ones asserting the cache *declines* to
answer.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.cache.embedder import HashingEmbedder, cosine
from gateway.cache.semantic import SemanticCache
from gateway.providers.mock import MockBehaviour, MockProvider
from gateway.reliability.executor import ReliableExecutor
from gateway.service import GatewayService
from gateway.types import ChatRequest, ChatResponse, Message, Usage


def req(text: str, *, temperature: float = 0.0, model: str | None = None) -> ChatRequest:
    return ChatRequest(
        messages=(Message(role="user", content=text),),
        temperature=temperature,
        model=model,
    )


def resp(text: str = "answer") -> ChatResponse:
    return ChatResponse(
        content=text,
        model="m",
        provider="p",
        usage=Usage(prompt_tokens=10, completion_tokens=20),
    )


# --- Embedder --------------------------------------------------------------


def test_embeddings_are_unit_length() -> None:
    vec = HashingEmbedder().embed("what is a circuit breaker")
    assert cosine(vec, vec) == pytest.approx(1.0)


def test_word_order_changes_the_embedding() -> None:
    """Bag-of-words alone would embed these identically.

    "is X better than Y" and "is Y better than X" are opposite questions. The
    bigram features exist precisely to stop that false match.
    """
    e = HashingEmbedder()
    a = e.embed("is redis better than postgres")
    b = e.embed("is postgres better than redis")
    assert cosine(a, b) < 0.99


def test_unrelated_prompts_are_not_similar() -> None:
    e = HashingEmbedder()
    a = e.embed("how do I configure a retry budget")
    b = e.embed("what is the capital of France")
    assert cosine(a, b) < 0.5


def test_empty_text_does_not_crash() -> None:
    assert HashingEmbedder().embed("") == [0.0] * 512


# --- Cache correctness -----------------------------------------------------


def test_identical_prompt_hits() -> None:
    cache = SemanticCache(embedder=HashingEmbedder())
    cache.store(req("what is hedging"), resp("hedging explained"))

    hit = cache.lookup(req("what is hedging"))
    assert hit is not None
    assert hit.cached is True
    assert hit.content == "hedging explained"
    assert hit.cache_similarity == pytest.approx(1.0)


def test_unrelated_prompt_misses() -> None:
    cache = SemanticCache(embedder=HashingEmbedder())
    cache.store(req("what is hedging"), resp())

    assert cache.lookup(req("how do I bake sourdough bread")) is None
    assert cache.stats.misses == 1


def test_different_model_never_matches() -> None:
    """Namespacing: same text, different model is a different question."""
    cache = SemanticCache(embedder=HashingEmbedder())
    cache.store(req("hello", model="gpt-4"), resp())

    assert cache.lookup(req("hello", model="claude")) is None


def test_different_temperature_never_matches() -> None:
    cache = SemanticCache(embedder=HashingEmbedder())
    cache.store(req("hello", temperature=0.0), resp())

    assert cache.lookup(req("hello", temperature=0.2)) is None


def test_high_temperature_bypasses_cache_entirely() -> None:
    """A caller asking for variety should not be served a stored answer."""
    cache = SemanticCache(embedder=HashingEmbedder(), max_temperature=0.3)

    cache.store(req("write me a poem", temperature=1.5), resp())
    assert len(cache) == 0

    assert cache.lookup(req("write me a poem", temperature=1.5)) is None
    assert cache.stats.skipped == 1
    assert cache.stats.lookups == 0


def test_threshold_is_respected() -> None:
    """At threshold 1.0 only an exact vector match may hit."""
    strict = SemanticCache(embedder=HashingEmbedder(), threshold=1.0)
    strict.store(req("what is a retry budget"), resp())

    assert strict.lookup(req("what is a retry budget please explain")) is None
    assert strict.lookup(req("what is a retry budget")) is not None


def test_hit_does_not_leak_the_stored_object() -> None:
    """Callers must not be able to mutate a cached entry for everyone else."""
    cache = SemanticCache(embedder=HashingEmbedder())
    cache.store(req("q"), resp("original"))

    first = cache.lookup(req("q"))
    assert first is not None
    first.content = "mutated"

    second = cache.lookup(req("q"))
    assert second is not None
    assert second.content == "original"


def test_lru_eviction_bounds_memory() -> None:
    cache = SemanticCache(embedder=HashingEmbedder(), max_entries=10)
    for i in range(25):
        cache.store(req(f"prompt number {i}"), resp())

    assert len(cache) == 10
    assert cache.stats.evictions == 15


def test_stats_track_tokens_saved() -> None:
    cache = SemanticCache(embedder=HashingEmbedder())
    cache.store(req("q"), resp())
    cache.lookup(req("q"))
    cache.lookup(req("q"))

    assert cache.stats.hits == 2
    assert cache.stats.tokens_saved == 60  # 2 hits x 30 tokens
    assert cache.stats.hit_rate == 1.0


# --- Service integration ---------------------------------------------------


async def test_cache_prevents_second_upstream_call() -> None:
    provider = MockProvider("p", MockBehaviour(base_latency_s=0.0, tail_probability=0.0))
    service = GatewayService(
        ReliableExecutor([provider]), SemanticCache(embedder=HashingEmbedder())
    )

    first = await service.complete(req("what is hedging"))
    second = await service.complete(req("what is hedging"))

    assert first.cached is False
    assert second.cached is True
    assert provider.call_count == 1


async def test_single_flight_collapses_concurrent_duplicates() -> None:
    """50 identical concurrent requests on a cold cache = 1 upstream call.

    Without this, caching provides no protection at the exact moment it is
    needed most: everyone misses simultaneously and stampedes the provider.
    """
    provider = MockProvider("p", MockBehaviour(base_latency_s=0.1, tail_probability=0.0))
    service = GatewayService(
        ReliableExecutor([provider]), SemanticCache(embedder=HashingEmbedder())
    )

    results = await asyncio.gather(*(service.complete(req("same question")) for _ in range(50)))

    assert len(results) == 50
    assert provider.call_count == 1


async def test_single_flight_propagates_failure_to_all_waiters() -> None:
    """Waiters must see the error, not hang forever."""
    provider = MockProvider("p", MockBehaviour.down())
    service = GatewayService(ReliableExecutor([provider], max_retries=0), None)

    results = await asyncio.gather(
        *(service.complete(req("q")) for _ in range(10)), return_exceptions=True
    )

    assert len(results) == 10
    assert all(isinstance(r, Exception) for r in results)


async def test_inflight_map_is_emptied() -> None:
    """A leaked entry would become an unbounded cache that never expires."""
    provider = MockProvider("p", MockBehaviour(base_latency_s=0.0, tail_probability=0.0))
    service = GatewayService(ReliableExecutor([provider]), None)

    await asyncio.gather(*(service.complete(req(f"q{i}")) for i in range(20)))

    assert service._inflight == {}  # noqa: SLF001


# --- Entity guard ----------------------------------------------------------
# Regression tests for the failure the traffic replay exposed: prompts that
# differ only by an identifier are ~99% similar under any embedder, so no
# threshold separates them. They must be separated structurally instead.


def test_different_identifiers_never_match() -> None:
    cache = SemanticCache(embedder=HashingEmbedder(), threshold=0.5)
    cache.store(req("restart service 41"), resp("restarted 41"))

    # Threshold is deliberately set low enough that similarity alone WOULD
    # have served this. Only the entity guard prevents it.
    assert cache.lookup(req("restart service 87")) is None


def test_same_identifier_still_matches() -> None:
    """The guard must not block legitimate paraphrases of the same entity."""
    cache = SemanticCache(embedder=HashingEmbedder(), threshold=0.7)
    cache.store(req("what is the status of order 4471"), resp("shipped"))

    hit = cache.lookup(req("what is the status of order 4471?"))
    assert hit is not None
    assert hit.content == "shipped"


def test_entity_signature_is_order_sensitive() -> None:
    sig = SemanticCache._entity_signature  # noqa: SLF001
    assert sig("move 10 to 20") != sig("move 20 to 10")
    assert sig("no digits here") == ""


def test_namespace_prefix_cannot_collide() -> None:
    """Namespace "...|1" must not prefix-match a key for entity 12.

    Both prompts are otherwise identical, so a naive `startswith` check would
    compare them and serve entity 1's answer for entity 12.
    """
    cache = SemanticCache(embedder=HashingEmbedder(), threshold=0.5)
    cache.store(req("check service 1"), resp("answer for 1"))

    assert cache.lookup(req("check service 12")) is None
