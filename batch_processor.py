import asyncio
import torch
from dataclasses import dataclass
from typing import List

from metrics import REQUEST_COUNTER, REQUEST_LATENCY, batch_size_histogram, queue_wait_time_histogram



# Represents a single generation request flowing through the system
@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    future: asyncio.Future  # where the result will be returned
    enqueue_time: float    # timestamp when request enters the queue


# Global async queue holding incoming requests
request_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()


async def enqueue_request(prompt: str, max_new_tokens: int):
    """
    Called by the FastAPI endpoint.
    Enqueues a request and waits until the batch worker resolves it.
    """
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    req = GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        future=future,
        enqueue_time=asyncio.get_event_loop().time() # timestamp when request enters the queue
    )

    await request_queue.put(req)

    # Wait until batch_worker sets the result
    return await future

# max_wait_time_ms is 500 for testing, bring back to 20 after
async def batch_worker(model, tokenizer, batch_size: int = 1, max_wait_ms: int = 500):
    """
    Background task:
    - Pulls requests from the queue
    - Batches them for a single GPU forward pass
    - Resolves each request's future
    - Records Prometheus metrics
    """
    while True:
        batch: List[GenerationRequest] = []

        # Wait for at least one request
        req = await request_queue.get()
        batch.append(req)

        start_time = asyncio.get_event_loop().time()

        # Try to fill the batch until batch_size or max_wait_ms is reached
        while len(batch) < batch_size:
            elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            remaining_ms = max_wait_ms - elapsed_ms
            if remaining_ms <= 0:
                break
            try:
                next_req = await asyncio.wait_for(request_queue.get(), timeout=remaining_ms / 1000)
                batch.append(next_req)
            except asyncio.TimeoutError:
                break  # time window exceeded, process current batch

        # Measure queue wait time for all requests
        now = asyncio.get_event_loop().time()
        for r in batch:
            wait_time = now - r.enqueue_time
            queue_wait_time_histogram.observe(wait_time)
            print(f"Request waited {wait_time:.3f}s in queue")

        # Record batch size
        batch_size_histogram.observe(len(batch))

        # Prepare batched input
        prompts = [r.prompt for r in batch]
        max_tokens = max(r.max_new_tokens for r in batch)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        # Run inference
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and set results for each request
        input_lengths = [len(tokenizer(r.prompt)["input_ids"]) for r in batch]
        for i, r in enumerate(batch):
            generated_ids = output[i][input_lengths[i]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            text = text.lstrip("\n ").rstrip()
            r.future.set_result(text)

        
        
