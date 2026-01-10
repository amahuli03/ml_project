# Optimization: Worker oversubscription study
# Run multiple concurrent generation calls to observe GPU contention
# Too many workers hurt performance due to GPU contention

import asyncio
from model import load_model
from metrics import measure_generation, save_results

PROMPT = "A slow jazz piano progression"
MAX_NEW_TOKENS = 128
WORKER_COUNTS = [1, 2, 4, 8]

model, tokenizer = load_model()
model.eval()

async def run_workers(n):
    tasks = [
        asyncio.to_thread(
            measure_generation,
            model,
            tokenizer,
            PROMPT,
            1,
            MAX_NEW_TOKENS,
        )
        for _ in range(n)
    ]
    return await asyncio.gather(*tasks)

results = {}

for workers in WORKER_COUNTS:
    metrics = asyncio.run(run_workers(workers))
    results[f"workers_{workers}"] = metrics

save_results("worker_oversubscription", results)
