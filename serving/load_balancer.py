# load_balancer.py
import asyncio

class RoundRobinLoadBalancer:
    def __init__(self):
        self.workers = []
        self.next_index = 0
        self.lock = asyncio.Lock()

    def register_worker(self, worker_callable):
        """
        Add a worker to the pool.
        worker_callable: async function that takes (prompt, max_new_tokens) and returns output
        """
        self.workers.append(worker_callable)

    async def route_request(self, prompt: str, max_new_tokens: int):
        """
        Send the request to the next worker in round-robin order.
        """
        async with self.lock:
            if not self.workers:
                raise RuntimeError("No workers registered in load balancer")

            worker = self.workers[self.next_index]
            self.next_index = (self.next_index + 1) % len(self.workers)

        # Call the worker
        return await worker(prompt, max_new_tokens)
