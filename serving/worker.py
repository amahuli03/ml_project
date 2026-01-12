# serving/worker.py
import asyncio
import torch
from batch_processor import enqueue_request
from metrics import WORKER_REQUEST_COUNTER

class Worker:
    def __init__(self, model, tokenizer, worker_id):
        self.model = model
        self.tokenizer = tokenizer
        self.worker_id = str(worker_id)

    async def handle_request(self, prompt: str, max_new_tokens: int):
        """
        Called by load balancer.
        Checks cache via batch_processor queue.
        Returns generated text.
        """
        # Increment the global counter with the worker_id label
        WORKER_REQUEST_COUNTER.labels(worker_id=self.worker_id).inc()
        # In current system, enqueue_request handles batching
        result = await enqueue_request(prompt, max_new_tokens)
        return result
