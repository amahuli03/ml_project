from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import asyncio
import time
from serving.cache import InMemoryCache


from model import load_model
from batch_processor import enqueue_request, batch_worker  # your batching code
import batch_processor

app = FastAPI(title="LLM Inference API with Batching")

# Load the model on startup
@app.on_event("startup")
def startup_event():
    print("Loading model on startup...")

    model, tokenizer = load_model()
    model.eval()

    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.cache = InMemoryCache()

    # 🔑 Pass model + tokenizer into the batch worker
    asyncio.create_task(
        batch_worker(
            model=app.state.model,
            tokenizer=app.state.tokenizer,
            batch_size=1,
            max_wait_ms=20,
        )
    )

    print("Model loaded and batch worker started.")


@app.get("/health")
def health():
    return {"status": "ok"}

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64

@app.post("/generate")
async def generate(req: GenerateRequest):
    cache = app.state.cache

    # Check cache
    cached = await cache.get(req.prompt, req.max_new_tokens)
    if cached is not None:
        return {
            "output": cached,
            "cache_hit": True,
        }

    # Cache miss → batch
    result = await enqueue_request(req.prompt, req.max_new_tokens)

    # Store result
    await cache.set(req.prompt, req.max_new_tokens, result)

    return {
        "output": result,
        "cache_hit": False,
    }


# Middleware to report request latency
@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    response.headers["X-Request-Latency"] = f"{latency:.6f}s"
    return response
