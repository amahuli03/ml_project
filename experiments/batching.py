import torch
from model import load_model
from metrics import measure_generation   
from metrics import save_results

PROMPT = "Once upon a time, in a distant future,"
MAX_NEW_TOKENS = 128

model, tokenizer = load_model(dtype=torch.float16)

for batch_size in [1, 2, 4, 8]:
    metrics = measure_generation(
        model,
        tokenizer,
        PROMPT,
        batch_size=batch_size,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        autocast_dtype=torch.float16,
    )

    save_results(f"batch_size_{batch_size}", metrics)
    print(f"Batch size {batch_size}: {metrics}")
