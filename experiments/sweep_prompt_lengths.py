import json5
import torch
from model import load_model
from metrics import measure_generation
from metrics import save_results

# -----------------------------
# CONFIG
# -----------------------------
PROMPT_BASE = "Once upon a time, in a distant future, "
MAX_NEW_TOKENS = 128  # number of tokens to generate after prompt
INPUT_LENGTHS = [16, 64, 128, 256]  # length of prompt in tokens
RESULTS_PATH = "../results/sweep_prompt_lengths.json"

BATCH_SIZE = 1
USE_CACHE = True
AUTOMATIC_MIXED_PRECISION = torch.float16

# -----------------------------
# LOAD MODEL
# -----------------------------
model, tokenizer = load_model(dtype=torch.float16)
model.eval()

# Warm-up (important for torch)
measure_generation(
    model,
    tokenizer,
    PROMPT_BASE,
    batch_size=1,
    max_new_tokens=16,
    use_cache=USE_CACHE,
    autocast_dtype=AUTOMATIC_MIXED_PRECISION,
)

# -----------------------------
# RUN SWEEP
# -----------------------------
results = []

for length in INPUT_LENGTHS:
    # Create a prompt of approximately 'length' tokens
    # This uses simple repetition to reach target length
    prompt_text = (PROMPT_BASE * ((length // len(PROMPT_BASE)) + 1))[:length]

    metrics = measure_generation(
        model,
        tokenizer,
        prompt_text,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=USE_CACHE,
        autocast_dtype=AUTOMATIC_MIXED_PRECISION,
    )

    # Record input length with metrics
    metrics["input_length"] = length
    results.append(metrics)

    print(f"Prompt length {length} tokens: {metrics}")

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_results("sweep_prompt_lengths", results)
print(metrics)