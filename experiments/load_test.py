# will simulate multiple concurrent clients hitting /generate and record timings

import asyncio
import aiohttp
import time
import json

URL = "http://localhost:8000/generate"  # your FastAPI endpoint
NUM_CLIENTS = 10  # concurrent clients
NUM_REQUESTS_PER_CLIENT = 5  # requests per client
MAX_NEW_TOKENS = 20

results = []

async def send_request(session, prompt):
    start = time.perf_counter()
    async with session.post(
        URL,
        json={"prompt": prompt, "max_new_tokens": MAX_NEW_TOKENS},
    ) as resp:
        data = await resp.json()
    latency = time.perf_counter() - start
    return {"prompt": prompt, "latency": latency, "cache_hit": data.get("cache_hit", False)}

async def client_task(client_id):
    async with aiohttp.ClientSession() as session:
        for i in range(NUM_REQUESTS_PER_CLIENT):
            prompt = f"Prompt {i % 3}"  # 3 repeating prompts to generate cache hits
            res = await send_request(session, prompt)
            results.append(res)
            # no sleep to create bursts

async def main():
    tasks = [asyncio.create_task(client_task(i)) for i in range(NUM_CLIENTS)]
    await asyncio.gather(*tasks)

    # Save results to JSON for later analysis
    with open("results/load_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Simple stats print
    latencies = [r["latency"] for r in results]
    print(f"Total requests: {len(results)}")
    print(f"Avg latency: {sum(latencies)/len(latencies):.3f}s")
    print(f"Min latency: {min(latencies):.3f}s, Max latency: {max(latencies):.3f}s")
    cache_hits = sum(r["cache_hit"] for r in results)
    print(f"Cache hits: {cache_hits}, Cache misses: {len(results)-cache_hits}")

if __name__ == "__main__":
    asyncio.run(main())