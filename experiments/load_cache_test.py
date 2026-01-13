# experiments/load_cache_test.py

import asyncio
import aiohttp
import time
import json
import os
from random import choice

# -----------------------------
# CONFIG
# -----------------------------
FASTAPI_URL = "http://localhost:8000/generate"  # your running FastAPI service
RESULTS_FILE = os.path.join("results", "cache_test_results.json")
os.makedirs("results", exist_ok=True)

NUM_REQUESTS = 50   # total number of requests to send
REPEAT_PROB = 0.6   # probability of sending a repeated prompt
PROMPTS = [
    "Hello world",
    "Once upon a time",
    "Write a poem about AI",
    "Explain the theory of relativity",
    "Summarize machine learning",
]

MAX_NEW_TOKENS = 20

# -----------------------------
# HELPER: send single request
# -----------------------------
async def send_request(session, prompt):
    start = time.perf_counter()
    async with session.post(
        FASTAPI_URL,
        json={"prompt": prompt, "max_new_tokens": MAX_NEW_TOKENS},
    ) as response:
        rjson = await response.json()
        latency_ms = (time.perf_counter() - start) * 1000
        cache_hit = rjson.get("cache_hit", False)
        return {
            "prompt": prompt,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "timestamp": time.time(),
        }

# -----------------------------
# MAIN TEST FUNCTION
# -----------------------------
async def main():
    results = []
    seen_prompts = []

    async with aiohttp.ClientSession() as session:
        for i in range(NUM_REQUESTS):
            # Decide whether to repeat a previous prompt
            if seen_prompts and (choice([True, False, False]) if REPEAT_PROB < 1 else True):
                prompt = choice(seen_prompts)
            else:
                prompt = choice(PROMPTS)
                seen_prompts.append(prompt)

            result = await send_request(session, prompt)
            results.append(result)
            print(f"Request {i+1}/{NUM_REQUESTS}: prompt='{prompt[:20]}...', "
                  f"latency={result['latency_ms']:.1f}ms, cache_hit={result['cache_hit']}")

            # Small delay to simulate realistic traffic
            await asyncio.sleep(0.05)

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to {RESULTS_FILE}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
