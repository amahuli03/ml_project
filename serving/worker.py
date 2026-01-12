# serving/worker.py
import asyncio
import torch
from batch_processor import enqueue_request

class Worker:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    async def handle_request(self, prompt: str, max_new_tokens: int):
        """
        Called by load balancer.
        Checks cache via batch_processor queue.
        Returns generated text.
        """
        # In your current system, enqueue_request handles batching
        result = await enqueue_request(prompt, max_new_tokens)
        return result
