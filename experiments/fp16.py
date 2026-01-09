import torch
from model import load_model
from metrics import measure_generation
from metrics import save_results

PROMPT = "Once upon a time, in a distant future,"
MAX_NEW_TOKENS = 256

model, tokenizer = load_model(dtype=torch.float16)

metrics = measure_generation(
    model,
    tokenizer,
    PROMPT,
    batch_size=1,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    autocast_dtype=torch.float16,
)
save_results("fp16", metrics)
print(metrics)
