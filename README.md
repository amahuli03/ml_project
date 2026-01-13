# GPU Inference Optimization — Practical LLM Serving Experiments

This project implements an LLM inference service with GPU batching,
TTL-based caching, and full observability. Autoscaling logic is implemented at the
application control-plane level, with infrastructure scaling simulated due to
single-GPU runtime constraints.

### What this project demonstrates
- Request batching via a single GPU forward pass to improve throughput.
- A lightweight in-memory cache to reduce repeated work and measure cache effects on latency.
- A round-robin load balancer and logical workers to experiment with concurrency/oversubscription.
- A simple queue-driven autoscaler that adjusts logical workers based on queue depth.
- End-to-end Prometheus metrics collection and basic experiment scripts for load testing and visualization.

### Quick architecture summary
- FastAPI exposes a single POST /generate endpoint which checks cache, then enqueues cache-miss requests.
- `batch_processor.batch_worker` pulls requests from a global asyncio queue, forms batches (configurable size and max-wait window), runs a GPU generation, and resolves each request's future.
- `serving/worker.py` provides logical worker handlers registered with the `RoundRobinLoadBalancer`.
- `serving/autoscaler.py` observes `request_queue` depth and adds/removes logical workers, exposing a Prometheus gauge for active workers.
- `metrics.py` defines Prometheus metrics for requests, latency, batch sizes, queue wait times, cache hits/misses, and worker activity.

## Quickstart:

Prerequisites:
- Python 3.10+
- `pip` or `conda`
- NVIDIA GPU with CUDA support

Steps:
1. Create a Python environment and install dependencies:
   - pip install -r requirements.txt
2. Run the app locally (development):
   - uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
3. Load tests and visualizations live in `experiments/`:
   - python experiments/load_cache_test.py
   - python experiments/load_test.py
   - python experiments/visualize_cache_results.py
4. Metrics: Prometheus metrics are exposed by the app (default port 8002) and can be plotted on Grafana (port 9090).

Configuration tips
- Model loading is in `model.load_model()`; by default the model is moved to CUDA. For CPU-only testing, remove `.cuda()`.
- Tune `batch_worker(...)` parameters (`batch_size`, `max_wait_ms`) to trade throughput for latency.
- Adjust `NUM_WORKERS` and autoscaler settings in `serving/app.py` to study oversubscription effects.

### Key Outcomes
- End-to-end inference under load
  - A FastAPI service handled concurrent client requests while batching, caching, and generating model outputs.
- Real, production-style metrics
  - The system exports Prometheus metrics for:
    - Request queue depth
    - Queue wait time
    - Batch size distribution
    - Per-worker request counts
    - Autoscaler decisions
- Observable autoscaling behavior
  - As request concurrency increased:
    - Queue depth and wait time rose predictably
    - The autoscaler computed higher desired worker counts
    - Scaling decisions were visible and explainable in Grafana
- Controlled performance tradeoffs
  - By tuning batch size limits and queue wait thresholds, I observed clear tradeoffs between:
    - Latency vs throughput
    - Batch efficiency vs tail latency
    - Cache effectiveness vs compute utilization

## License
This project is released under the MIT License. See the bundled `LICENSE` file for the full text and permissions.
