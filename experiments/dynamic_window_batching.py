# Optimization: Time-window dynamic batching
# Dynamic batching is a technique where inference requests arriving within a short time window 
# (e.g. 5–20ms) are coalesced into a single batch before being sent to the model.
# Collect requests for a short time window, then batch them together

import time
from model import load_model
from metrics import measure_generation, save_results

PROMPT = "A soft ambient piano melody"
WINDOW_MS = 20
TOTAL_REQUESTS = 16
MAX_NEW_TOKENS = 128

model, tokenizer = load_model()
model.eval()

batch = []
start_times = []
results = []

start_window = time.time()

for i in range(TOTAL_REQUESTS):
    batch.append(PROMPT)
    start_times.append(time.time())

    elapsed_ms = (time.time() - start_window) * 1000
    if elapsed_ms >= WINDOW_MS or i == TOTAL_REQUESTS - 1:
        metrics = measure_generation(
            model,
            tokenizer,
            PROMPT,
            batch_size=len(batch),
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )

        for st in start_times:
            record = metrics.copy()
            record["queue_latency_s"] = time.time() - st
            results.append(record)

        batch = []
        start_times = []
        start_window = time.time()

save_results("dynamic_window_batching", results)
print(results)