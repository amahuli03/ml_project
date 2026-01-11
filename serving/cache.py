import asyncio
from typing import Dict, Tuple

class InMemoryCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, int], str] = {}
        self._lock = asyncio.Lock()

    async def get(self, prompt: str, max_new_tokens: int):
        key = (prompt, max_new_tokens)
        async with self._lock:
            return self._cache.get(key)

    async def set(self, prompt: str, max_new_tokens: int, value: str):
        key = (prompt, max_new_tokens)
        async with self._lock:
            self._cache[key] = value
