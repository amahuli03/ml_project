# serving/cache.py
import asyncio
import time
from typing import Dict, Tuple, Optional
from prometheus_client import Gauge, Counter

# Prometheus metric for cache size
CACHE_SIZE = Gauge("llm_cache_size", "Current number of entries in the cache")

class InMemoryCache:
    def __init__(self, ttl_seconds: int = 300):
        """
        ttl_seconds: time-to-live for cache entries (default: 5 minutes)
        """
        self._cache: Dict[Tuple[str, int], Tuple[str, float]] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds

        # Track hits/misses (optional, already in metrics.py)
        # You could remove these if you want to reuse CACHE_HITS / CACHE_MISSES
        # self.hits = 0
        # self.misses = 0

    async def get(self, prompt: str, max_new_tokens: int) -> Optional[str]:
        """
        Retrieve a cached value if it exists and hasn't expired.
        """
        key = (prompt, max_new_tokens)
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                value, timestamp = entry
                # Check TTL
                if time.time() - timestamp < self._ttl:
                    return value
                else:
                    # Expired → remove entry
                    del self._cache[key]
                    CACHE_SIZE.set(len(self._cache))
        return None

    async def set(self, prompt: str, max_new_tokens: int, value: str):
        """
        Store a value in the cache with a timestamp.
        """
        key = (prompt, max_new_tokens)
        async with self._lock:
            self._cache[key] = (value, time.time())
            CACHE_SIZE.set(len(self._cache))

    async def cleanup(self):
        """
        Remove expired entries.
        Can be called periodically as a background task.
        """
        async with self._lock:
            now = time.time()
            expired_keys = [k for k, (_, ts) in self._cache.items() if now - ts >= self._ttl]
            for k in expired_keys:
                del self._cache[k]
            CACHE_SIZE.set(len(self._cache))

    async def start_periodic_cleanup(self, interval_seconds: int = 60):
        """
        Start a background task that cleans up expired entries every interval_seconds.
        """
        while True:
            await asyncio.sleep(interval_seconds)
            await self.cleanup()
