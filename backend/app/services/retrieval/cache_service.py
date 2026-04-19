"""
In-memory query response cache.

Caches final LLM responses keyed by (query, provider) to avoid repeated
API calls for identical questions.  Entries expire after CACHE_TTL_SECONDS.

Thread safety: Python's GIL makes dict reads/writes atomic at the bytecode
level, which is sufficient for this MVP.  No locks needed.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS: int = 15 * 60  # 15 minutes

_cache: dict[str, dict] = {}


def _make_key(query: str, provider: str) -> str:
    """Build a cache key from the query text and LLM provider."""
    return f"{query.strip().lower()}_{provider.strip().lower()}"


def is_cache_valid(entry: dict) -> bool:
    """Check whether a cache entry is still within its TTL."""
    return (time.time() - entry["timestamp"]) < CACHE_TTL_SECONDS


def get_cache(query: str, provider: str) -> dict | None:
    """
    Return the cached response for (query, provider), or None on miss/expiry.
    """
    key = _make_key(query, provider)
    entry = _cache.get(key)

    if entry is None:
        return None

    if not is_cache_valid(entry):
        del _cache[key]
        logger.info("Cache expired for key '%.60s…'", key)
        return None

    logger.info("Cache hit for key '%.60s…'", key)
    return entry["response"]


def set_cache(query: str, provider: str, response: dict) -> None:
    """Store a response in the cache with the current timestamp."""
    key = _make_key(query, provider)
    _cache[key] = {
        "response": response,
        "timestamp": time.time(),
    }
    logger.info("Cache set for key '%.60s…'", key)


def clear_cache() -> None:
    """Wipe the entire cache.  Useful for tests."""
    _cache.clear()
    logger.info("Cache cleared.")


def get_cache_stats() -> dict:
    """Return basic cache statistics."""
    now = time.time()
    valid = sum(1 for e in _cache.values() if is_cache_valid(e))
    return {
        "total_entries": len(_cache),
        "valid_entries": valid,
        "expired_entries": len(_cache) - valid,
        "ttl_seconds": CACHE_TTL_SECONDS,
    }
