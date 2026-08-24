"""Semantic response cache.

An exact-match cache on prompt text almost never hits: change one word and you
pay full price again. A semantic cache embeds the prompt and serves a stored
response when a previous prompt was close enough in vector space.

The danger is obvious and worth stating plainly: **a false hit returns the
wrong answer to the user.** That is a correctness bug, not a performance
regression, and it is strictly worse than a cache miss. Everything below is
shaped by that asymmetry -- when in doubt, miss.

Three defences:

1. **Namespacing.** Only requests that are actually interchangeable can match.
   Different model, temperature, or token limit means a different namespace,
   never a candidate.
2. **A conservative threshold**, calibrated against a labelled set rather than
   guessed. See `chaos/cache_bench.py`.
3. **A temperature ceiling.** A caller asking for temperature 1.5 wants
   variety; serving them a stored answer defeats the point of the request.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import structlog

from gateway.cache.embedder import Embedder, cosine, default_embedder
from gateway.types import ChatRequest, ChatResponse

log = structlog.get_logger(__name__)

_DIGITS = re.compile(r"\d+")


@dataclass(slots=True)
class CacheEntry:
    vector: list[float]
    response: ChatResponse
    prompt: str
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass(slots=True)
class CacheStats:
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    skipped: int = 0
    """Requests never considered for caching, e.g. temperature above ceiling."""

    tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class SemanticCache:
    """An in-memory vector cache with LRU eviction.

    Lookup is a brute-force scan over the namespace: O(n) in entries. At the
    default 10k-entry cap that is well under a millisecond and costs nothing in
    dependencies or operational surface. A production deployment with millions
    of entries wants an ANN index (pgvector/HNSW) -- but adopting one here would
    trade a real dependency for a speedup on a scan that isn't the bottleneck.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        threshold: float | None = None,
        max_entries: int = 10_000,
        max_temperature: float = 0.3,
    ) -> None:
        self.embedder = embedder or default_embedder()
        # Fall back to the embedder's calibrated threshold rather than a global
        # constant, since the two backends score similarity on different scales.
        self.threshold = (
            threshold if threshold is not None else self.embedder.recommended_threshold
        )
        self.max_entries = max_entries
        self.max_temperature = max_temperature
        self.stats = CacheStats()
        # OrderedDict gives O(1) LRU: move_to_end on access, popitem(last=False)
        # to evict the coldest entry.
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

        log.info("cache.ready", embedder=self.embedder.name, threshold=self.threshold)

    # -- keying -------------------------------------------------------------

    @staticmethod
    def _entity_signature(text: str) -> str:
        """Identifiers in the prompt, matched EXACTLY rather than fuzzily.

        This is the most important safeguard in the cache, and it exists
        because the evaluation caught the failure it prevents.

        "restart service 41" and "restart service 87" are ~99% similar under any
        embedder -- they differ by two characters out of eighteen. No similarity
        threshold separates them without also rejecting every genuine
        paraphrase, because the distinguishing token is precisely the one that
        carries the least lexical weight. Raising the threshold to catch them
        just destroys recall while still leaking some through.

        So identifiers are pulled out of the fuzzy comparison entirely and made
        part of the namespace. Prompts about different entities can never be
        compared, whatever their vectors say. Semantic matching for *wording*,
        exact matching for *entities*.
        """
        return ",".join(_DIGITS.findall(text))

    @classmethod
    def _namespace(cls, request: ChatRequest) -> str:
        """Only mutually substitutable requests share a namespace.

        Sampling parameters are part of the identity of a request. Two prompts
        with identical text but different temperatures are different questions
        as far as the caller is concerned.
        """
        entities = cls._entity_signature(request.cache_key_text())
        return (
            f"{request.model or 'default'}|{request.temperature}"
            f"|{request.max_tokens}|{entities}"
        )

    def _cacheable(self, request: ChatRequest) -> bool:
        return request.temperature <= self.max_temperature

    # -- API ----------------------------------------------------------------

    def lookup(self, request: ChatRequest) -> ChatResponse | None:
        """Return a stored response if one is close enough, else None."""
        if not self._cacheable(request):
            self.stats.skipped += 1
            return None

        self.stats.lookups += 1
        namespace = self._namespace(request)
        query = self.embedder.embed(request.cache_key_text())

        best_key: str | None = None
        best_score = -1.0
        prefix = f"{namespace}|"
        for key, entry in self._entries.items():
            # The trailing separator matters: without it namespace "...|1"
            # would prefix-match key "...|12|<hash>" and compare prompts about
            # entity 1 against prompts about entity 12.
            if not key.startswith(prefix):
                continue
            score = cosine(query, entry.vector)
            if score > best_score:
                best_score, best_key = score, key

        if best_key is None or best_score < self.threshold:
            self.stats.misses += 1
            return None

        entry = self._entries[best_key]
        self._entries.move_to_end(best_key)
        entry.hits += 1
        self.stats.hits += 1
        self.stats.tokens_saved += entry.response.usage.total_tokens

        log.info("cache.hit", similarity=round(best_score, 4), prompt=entry.prompt[:60])

        # Return a copy. Handing out the stored object would let a caller
        # mutate every future hit for that entry.
        return ChatResponse(
            content=entry.response.content,
            model=entry.response.model,
            provider=entry.response.provider,
            usage=entry.response.usage,
            cached=True,
            cache_similarity=best_score,
        )

    def store(self, request: ChatRequest, response: ChatResponse) -> None:
        if not self._cacheable(request):
            return

        prompt = request.cache_key_text()
        key = f"{self._namespace(request)}|{hash(prompt)}"
        self._entries[key] = CacheEntry(
            vector=self.embedder.embed(prompt),
            response=response,
            prompt=prompt,
        )
        self._entries.move_to_end(key)
        self.stats.stores += 1

        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
