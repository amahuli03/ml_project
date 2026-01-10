# GPU Inference Optimization for Generative Models

This project explores **inference-time optimization techniques** for autoregressive generative models on a single GPU.  
The goal is to understand how **model-level, kernel-level, and system-level optimizations** affect real-world inference metrics such as **time-to-first-token (TTFT)**, **throughput**, and **GPU memory usage**.

## Motivation

Most ML projects focus on training or model quality.  
In production systems, however, **inference performance and cost** are often the dominant concerns.

This project was designed to answer questions like:
- What limits GPU throughput during inference?
- How do batching and concurrency affect latency?
- When does autoscaling help — and when does it stop helping?
- Which optimizations improve throughput vs responsiveness?

---

## System Overview

- **Hardware:** Single NVIDIA GPU (RTX 2000 Ada)
- **Framework:** PyTorch + Hugging Face Transformers
- **Execution model:** Python-based inference with async concurrency
- **Focus:** Autoregressive generation performance

The system is structured as a lightweight inference harness with reusable metric collection and isolated experiment scripts.

---

## Metrics

All experiments measure the following inference metrics:

- **Time to First Token (TTFT):**  
  Time from request submission to the first generated token (user-perceived latency)

- **Latency:**  
  Total generation time for the request

- **Throughput (tokens/sec):**  
  GPU efficiency under load

- **Peak VRAM usage:**  
  Memory footprint during inference

These metrics reflect real production concerns such as **SLA compliance** and **cost efficiency**.

---

## Experiments

The following classes of inference optimizations were explored:

### Model-Level Optimizations
- KV cache reuse
- Mixed-precision (FP16) inference

### Kernel-Level Optimizations
- `torch.compile` for kernel fusion and reduced Python overhead

### System-Level Optimizations
- Static and dynamic batching
- Concurrency tuning and oversubscription analysis
- Latency-driven autoscaling simulation

### Capacity Planning
- Throughput saturation analysis
- GPU utilization limits under increasing batch size

Each experiment is isolated in the `experiments/` directory and writes structured JSON results to `results/`.

---

## Key Results & Insights

Some high-level observations from the experiments:

- **Batching is the largest throughput lever**, but increases TTFT
- **KV caching significantly reduces per-token latency** during generation
- **Concurrency has a sweet spot**; oversubscribing the GPU hurts both latency and throughput
- **Autoscaling improves latency under bursty load**, but benefits plateau on a single GPU
- **Throughput saturates well before maximum batch sizes**, highlighting the importance of capacity planning

These results illustrate the tradeoffs between **latency, throughput, and resource utilization** in real inference systems.


### Table 1: Core Inference Optimizations
| Optimization  | Latency (s) | TTFT (s) | Tokens/sec | Peak VRAM (MB) |
| ------------- | ----------- | -------- | ---------- | -------------- |
| Baseline      | 1.25        | 0.38     | 205.3      | 346.3          |
| FP16          | 1.29        | 0.45     | 198.5      | 179.3          |
| KV Cache      | 1.17        | 0.37     | 219.1      | 339.3          |
| Torch Compile | 1.24        | 0.0066   | 206.8      | 179.3          |

Table 1 shows that KV caching improves throughput and latency moderately, FP16 drastically reduces VRAM usage, and Torch compilation massively reduces time-to-first-token (TTFT). These are the main single-request inference optimizations.

### Table 2: Effect of Batch Size on Throughput and Latency
| Batch Size | Latency (s) | TTFT (s) | Tokens Generated | Tokens/sec | Peak VRAM (MB) |
| ---------- | ----------- | -------- | ---------------- | ---------- | -------------- |
| 1          | 1.25        | 0.38     | 256              | 205        | 346            |
| 2          | 0.62        | 0.021    | 256              | 414        | 180            |
| 4          | 0.62        | 0.007    | 512              | 826        | 186            |
| 8          | 0.62        | 0.0067   | 1024             | 1644       | 198            |

Throughput scales nearly linearly with batch size, while latency and memory usage increase minimally. This illustrates why batching is critical for efficient GPU utilization in LLM serving.

### Table 3: Impact of Worker Concurrency (Oversubscription)
| Workers | Avg Latency (s) | Avg TTFT (s) | Avg Tokens/sec | Avg Peak VRAM (MB) |
| ------- | --------------- | ------------ | -------------- | ------------------ |
| 1       | 0.59            | 0.40         | 215            | 334                |
| 2       | 1.24            | 0.022        | 103            | 349                |
| 4       | 4.74            | 0.055        | 27             | 376                |
| 8       | 10.81           | 0.11         | 11.8           | 432                |

Adding more workers to a single GPU beyond 1 dramatically reduces throughput and increases latency due to oversubscription and memory contention. Highlights why careful scheduling and batching is better than naive parallelism.

