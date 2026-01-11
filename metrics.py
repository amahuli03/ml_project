from prometheus_client import Histogram, Counter

# Prometheus metrics
REQUEST_COUNTER = Counter(
    "llm_generate_requests_total",
    "Total number of /generate requests"
)
REQUEST_LATENCY = Histogram(
    "llm_generate_request_latency_seconds",
    "Latency of /generate requests in seconds"
)

# Track batch sizes processed by the batch_worker
batch_size_histogram = Histogram(
    "llm_batch_size",
    "Number of requests processed in a batch"
)

# How long each request waited in the queue before being processed
queue_wait_time_histogram = Histogram(
    "llm_queue_wait_time_seconds",
    "Time each request spends waiting in the queue before being processed"
)