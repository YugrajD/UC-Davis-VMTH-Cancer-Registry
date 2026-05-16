"""In-memory TTL cache for read-only API responses."""

import functools
import hashlib
import json
import logging
from typing import Callable

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Registry of all cache instances for bulk invalidation
_caches: dict[str, TTLCache] = {}


def get_cache(namespace: str, maxsize: int, ttl: int) -> TTLCache:
    """Get or create a named cache instance."""
    if namespace not in _caches:
        _caches[namespace] = TTLCache(maxsize=maxsize, ttl=ttl)
    return _caches[namespace]


def clear_all_caches() -> None:
    """Clear every registered cache. Called after data ingestion."""
    for name, cache in _caches.items():
        cache.clear()
        logger.info("Cleared cache: %s", name)


def _make_cache_key(**kwargs) -> str:
    """Build a deterministic cache key from query parameters."""
    normalized = {}
    for k, v in sorted(kwargs.items()):
        if v is None:
            continue
        if isinstance(v, list):
            normalized[k] = tuple(sorted(v))
        else:
            normalized[k] = v
    raw = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()


def cached_response(namespace: str, maxsize: int = 256, ttl: int = 60):
    """Decorator that caches endpoint return values by query params.

    Usage:
        @router.get("/summary")
        @cached_response("dashboard_summary", ttl=60)
        async def get_summary(db = Depends(get_db)):
            ...

    The decorator inspects the function's keyword arguments (excluding
    the 'db' session) to build the cache key.
    """
    cache = get_cache(namespace, maxsize, ttl)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build key from query params only — exclude FastAPI injected objects
            key_params = {k: v for k, v in kwargs.items() if k not in ("db", "request")}
            key = _make_cache_key(**key_params)

            if key in cache:
                return cache[key]

            result = await func(*args, **kwargs)
            cache[key] = result
            return result

        return wrapper

    return decorator
