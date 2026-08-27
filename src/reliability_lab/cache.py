from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared utilities — use these in both ResponseCache and SharedRedisCache
# ---------------------------------------------------------------------------

PRIVACY_PATTERNS = re.compile(
    r"\b(balance|password|credit.card|ssn|social.security|user.\d+|account.\d+)\b",
    re.IGNORECASE,
)


def _is_uncacheable(query: str) -> bool:
    """Return True if query contains privacy-sensitive keywords."""
    return bool(PRIVACY_PATTERNS.search(query))


def _looks_like_false_hit(query: str, cached_key: str) -> bool:
    """Return True if query and cached key contain different 4-digit numbers (years, IDs)."""
    nums_q = set(re.findall(r"\b\d{4}\b", query))
    nums_c = set(re.findall(r"\b\d{4}\b", cached_key))
    return bool(nums_q and nums_c and nums_q != nums_c)


# ---------------------------------------------------------------------------
# In-memory cache (existing)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    key: str
    value: str
    created_at: float
    metadata: dict[str, str]


class ResponseCache:
    """In-memory, process-local response cache.

    Lookups are by n-gram cosine similarity rather than exact match, so
    paraphrased queries can still hit. Privacy-sensitive queries are never
    stored or served, and matches above the similarity threshold are still
    rejected if they look like a false hit (e.g. same shape, different year).
    For multi-instance deployments, use SharedRedisCache instead.
    """

    def __init__(self, ttl_seconds: int, similarity_threshold: float):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._entries: list[CacheEntry] = []
        self.false_hit_log: list[dict[str, object]] = []

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response by semantic similarity.

        Returns (value, score) on a usable hit, or (None, score) — where
        score is the best similarity found, possibly 0.0 — on a miss.
        """
        if _is_uncacheable(query):
            return None, 0.0

        now = time.time()
        self._entries = [e for e in self._entries if now - e.created_at <= self.ttl_seconds]

        best: CacheEntry | None = None
        best_score = 0.0
        for entry in self._entries:
            score = self.similarity(query, entry.key)
            if score > best_score:
                best, best_score = entry, score

        if best is None or best_score < self.similarity_threshold:
            return None, best_score

        if _looks_like_false_hit(query, best.key):
            self.false_hit_log.append(
                {
                    "query": query,
                    "cached_key": best.key,
                    "score": best_score,
                    "reason": "date_or_number_mismatch",
                }
            )
            return None, best_score

        return best.value, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response, unless the query is privacy-sensitive."""
        if not _is_uncacheable(query):
            self._entries.append(CacheEntry(query, value, time.time(), metadata or {}))

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Cosine similarity over word tokens plus character 3-grams.

        Character n-grams catch near-duplicate phrasing that plain word
        overlap (Jaccard) misses, e.g. minor typos or word-order changes.
        """
        if a == b:
            return 1.0

        def tokens(s: str) -> list[str]:
            s = s.lower()
            words = re.findall(r"\w+", s)
            grams = [s[i : i + 3] for i in range(max(0, len(s) - 2))]
            return words + grams

        vec_a, vec_b = Counter(tokens(a)), Counter(tokens(b))
        if not vec_a or not vec_b:
            return 0.0

        dot = sum(count * vec_b.get(token, 0) for token, count in vec_a.items())
        norm_a = math.sqrt(sum(count * count for count in vec_a.values()))
        norm_b = math.sqrt(sum(count * count for count in vec_b.values()))
        return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Redis shared cache (new)
# ---------------------------------------------------------------------------


class SharedRedisCache:
    """Redis-backed cache shared across gateway instances.

    Each entry is a Redis hash keyed by ``{prefix}{query_hash}`` with
    "query" and "response" fields and a TTL set via EXPIRE, so expiry is
    handled by Redis rather than manual eviction. Exact-hash lookups are
    O(1); paraphrase matches fall back to scanning keys under ``prefix``
    and scoring each cached query with ResponseCache.similarity().
    Applies the same privacy and false-hit guardrails as ResponseCache.
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        similarity_threshold: float,
        prefix: str = "rl:cache:",
    ):
        import redis as redis_lib

        self._redis_exceptions = redis_lib.exceptions
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.false_hit_log: list[dict[str, object]] = []
        self._redis: Any = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self._redis.ping())
        except self._redis_exceptions.RedisError:
            return False

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response from Redis.

        Tries an exact hash match first (score 1.0); falls back to a
        similarity scan over all entries under this cache's prefix.
        Returns (value, score) on a usable hit, or (None, score) on a miss.
        """
        if _is_uncacheable(query):
            return None, 0.0

        exact_key = f"{self.prefix}{self._query_hash(query)}"
        exact_response = self._redis.hget(exact_key, "response")
        if exact_response is not None:
            return exact_response, 1.0

        best_value: str | None = None
        best_key: str | None = None
        best_score = 0.0
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            cached_query = self._redis.hget(key, "query")
            if not cached_query:
                continue
            score = ResponseCache.similarity(query, cached_query)
            if score > best_score:
                best_value = self._redis.hget(key, "response")
                best_key = cached_query
                best_score = score

        if best_value is None or best_score < self.similarity_threshold:
            return None, best_score

        if _looks_like_false_hit(query, best_key or ""):
            self.false_hit_log.append(
                {
                    "query": query,
                    "cached_key": best_key,
                    "score": best_score,
                    "reason": "date_or_number_mismatch",
                }
            )
            return None, best_score

        return best_value, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in Redis with a TTL, unless privacy-sensitive."""
        if _is_uncacheable(query):
            return
        key = f"{self.prefix}{self._query_hash(query)}"
        self._redis.hset(key, mapping={"query": query, "response": value})
        self._redis.expire(key, self.ttl_seconds)

    def flush(self) -> None:
        """Remove all entries with this cache prefix (for testing)."""
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            self._redis.delete(key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            self._redis.close()

    @staticmethod
    def _query_hash(query: str) -> str:
        """Deterministic short hash for a query string."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]
