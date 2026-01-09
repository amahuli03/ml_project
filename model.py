import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(
    model_name="distilgpt2",
    dtype=torch.float32,
    compile_model=False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).cuda()
    model.eval()

    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")

    return model, tokenizer
