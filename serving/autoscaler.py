import asyncio
from metrics import queue_wait_time_histogram
from serving.worker import Worker
from model import load_model
from batch_processor import request_queue

class AutoScaler:
    def __init__(self, model, tokenizer, initial_workers=1, max_workers=4, min_workers=1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.workers = []
        self.worker_tasks = []

        # Start with initial_workers
        for i in range(initial_workers):
            self.add_worker()

    def add_worker(self):
        worker_id = len(self.workers)
        if worker_id >= self.max_workers:
            return
        worker = Worker(self.model, self.tokenizer, worker_id)
        self.workers.append(worker)
        # Update Prometheus gauge
        from metrics import ACTIVE_WORKERS
        ACTIVE_WORKERS.set(len(self.workers))
        print(f"[AutoScaler] Added worker {worker_id}")

    def remove_worker(self):
        if len(self.workers) <= self.min_workers:
            return
        worker = self.workers.pop()
        from metrics import ACTIVE_WORKERS
        ACTIVE_WORKERS.set(len(self.workers))
        print(f"[AutoScaler] Removed worker {worker.worker_id}")

    async def monitor(self, interval=1.0):
        while True:
            queue_size = request_queue.qsize()
            num_workers = len(self.workers)

            # Scaling logic (smaller thresholds for testing)
            if queue_size > 2 and num_workers < self.max_workers:
                self.add_worker()
            elif queue_size == 0 and num_workers > self.min_workers:
                self.remove_worker()

            print(f"[AutoScaler] Queue size: {queue_size}, Workers: {len(self.workers)}")
            await asyncio.sleep(interval)
