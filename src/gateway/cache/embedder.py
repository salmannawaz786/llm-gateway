"""Embedding backends for the semantic cache.

The cache is deliberately embedder-agnostic. Which model produces the vectors
is an implementation detail; what matters is that the cache can be *evaluated*
against whichever one you choose, because the safety of a semantic cache
depends entirely on embedding quality.

Two backends ship:

  HashingEmbedder      zero dependencies, lexical. Fast, deterministic, and
                       honest about what it is: it matches wording, not
                       meaning. Good enough for near-duplicate traffic
                       (retries, templated prompts) which is where most real
                       cache hits come from anyway.

  MiniLMEmbedder       genuine sentence embeddings via sentence-transformers.
                       Catches paraphrase. Optional, because it pulls ~2.5GB
                       of torch and the gateway must stay clone-and-run.

`chaos/cache_bench.py` measures precision and recall for whichever is
installed, so the difference is a number rather than a claim.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

# Function words dominate token counts while carrying little signal about what
# a prompt asks, so dropping them stops "what is the ..." from making two
# unrelated questions look similar.
#
# The list is deliberately MINIMAL -- only articles, copulas, conjunctions and
# prepositions. An earlier version also stripped question words ("what", "how",
# "why") and verbs like "explain" and "tell", which was a genuine correctness
# bug: "what is a retry budget" and "what is a retry budget please explain"
# reduced to identical token sets and scored 1.0 similarity. Question words
# carry intent -- "how to X" and "what is X" want different answers -- so an
# aggressive stoplist trades false hits for a marginal recall gain. Wrong side
# of that trade. There is a regression test for exactly this.
_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in",
    "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "with", "you", "your"
])


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a unit-length vector."""

    name: str
    dimensions: int
    recommended_threshold: float
    """Calibrated by `chaos/cache_bench.py`, not guessed.

    Different embedders put similarity on different scales, so a single global
    default cannot be correct for both. Each backend carries its own.
    """

    def embed(self, text: str) -> list[float]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity.

    Both embedders return L2-normalised vectors, so this reduces to a dot
    product -- but the general form is kept so a future embedder that forgets
    to normalise cannot silently corrupt every similarity score.
    """
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-width vector (the "hashing trick").

    Each token is hashed to a bucket and its weight added there. No vocabulary
    is stored, so the memory cost is fixed regardless of how much traffic flows
    through -- which is the property that makes this viable as a default.

    Word bigrams are included alongside unigrams so that word ORDER carries
    some signal. Without them "is X better than Y" and "is Y better than X"
    embed identically, which is exactly the kind of false cache hit that turns
    a cost optimisation into a correctness bug.

    This is lexical similarity, not semantic. It is labelled honestly and
    measured accordingly.
    """

    name = "hashing-v1"
    recommended_threshold = 0.72
    """Lowest threshold with 100% precision on the labelled set (recall 90%)."""

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = [w for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        return words + bigrams

    def _bucket(self, token: str) -> tuple[int, float]:
        """Map a token to (bucket, sign).

        The signed hashing trick: half the tokens contribute negatively. This
        makes collisions cancel out on average instead of always inflating
        similarity, which matters a lot at only 512 dimensions.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        tokens = self._tokens(text)
        if not tokens:
            return vec

        # Sublinear term-frequency weighting: a word repeated ten times is more
        # important than one used once, but not ten times more.
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1

        for tok, count in counts.items():
            bucket, sign = self._bucket(tok)
            vec[bucket] += sign * (1.0 + math.log(count))

        return _normalise(vec)


class MiniLMEmbedder:
    """Real sentence embeddings via `all-MiniLM-L6-v2`.

    Runs on CPU, ~80MB of weights, free. Loading is deferred to first use so
    importing this module never costs a model load.
    """

    name = "all-MiniLM-L6-v2"
    dimensions = 384
    recommended_threshold = 0.85

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = model.encode(text, normalize_embeddings=True)  # type: ignore[attr-defined]
        return [float(x) for x in vector]


def default_embedder() -> Embedder:
    """Prefer real embeddings when available, fall back to lexical.

    The fallback is what keeps `git clone && pytest` working without a 2.5GB
    download. The gateway logs which one it picked so a cache hit rate is never
    reported without the context needed to interpret it.
    """
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return HashingEmbedder()
    except Exception as exc:  # noqa: BLE001
        # Deliberately broad. An optional dependency that is *installed but
        # unusable* must degrade, not take the gateway down with it -- and it
        # fails in ways that are not ImportError. This machine hit
        # `OSError: [WinError 1114]` from torch failing to load `c10.dll`
        # (missing MSVC runtime), which the ImportError-only version of this
        # function let escape straight out of application startup.
        log.warning("embedder.optional_backend_unusable", error=str(exc), fallback="hashing-v1")
        return HashingEmbedder()
    return MiniLMEmbedder()
