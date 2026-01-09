import time
import torch
import json
import pathlib

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def measure_generation(
    model,
    tokenizer,
    prompt,
    batch_size=1,
    max_new_tokens=128,
    use_cache=True,
    autocast_dtype=None,
):
    # Prepare inputs
    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    torch.cuda.reset_peak_memory_stats()
    cuda_sync()

    # --- Measure TTFT (first token) ---
    start_ttft = time.perf_counter()
    if autocast_dtype:
        with torch.autocast("cuda", dtype=autocast_dtype):
            first_output = model.generate(
                **inputs,
                max_new_tokens=1,
                use_cache=use_cache,
            )
    else:
        first_output = model.generate(
            **inputs,
            max_new_tokens=1,
            use_cache=use_cache,
        )
    cuda_sync()
    ttft = time.perf_counter() - start_ttft

    # --- Measure remaining tokens ---
    start_total = time.perf_counter()
    if autocast_dtype:
        with torch.autocast("cuda", dtype=autocast_dtype):
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
    else:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )
    cuda_sync()
    total_elapsed = time.perf_counter() - start_total

    # Tokens generated (exclude prompt length)
    generated_tokens = (output.shape[-1] - inputs["input_ids"].shape[-1]) * batch_size

    return {
        "latency_s": total_elapsed,
        "time_to_first_token_s": ttft,
        "tokens_generated": generated_tokens,
        "tokens_per_sec": generated_tokens / total_elapsed,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }

def save_results(name, metrics):
    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {path}")
