# experiments/latency_autoscale.py
# Optimization: Latency-based autoscaling simulation
# How: Adjust worker count based on observed TTFT

from model import load_model
from metrics import measure_generation, save_results

PROMPT = "A cinematic orchestral swell"
MAX_NEW_TOKENS = 128
MAX_WORKERS = 4
MIN_WORKERS = 1

model, tokenizer = load_model()
model.eval()

workers = 1
results = []

for step in range(12):
    metrics = measure_generation(
        model,
        tokenizer,
        PROMPT,
        batch_size=workers,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
    )

    metrics["active_workers"] = workers
    results.append(metrics)

    # Autoscaling rule (simplified HPA)
    if metrics["time_to_first_token_s"] > 0.6 and workers < MAX_WORKERS:
        workers += 1
    elif metrics["time_to_first_token_s"] < 0.25 and workers > MIN_WORKERS:
        workers -= 1

save_results("latency_autoscale", results)
print(results)