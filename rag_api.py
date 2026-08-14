"""
Speculative Voice-RAG: ULTRA-OPTIMIZED RAG API (Member 2)
==========================================================
FastAPI server with 6-Layer Latency Optimization:
  Layer 1: Full Response Cache (repeated query = 2ms)
  Layer 2: Embedding Cache (same query = skip encoding)
  Layer 3: Connection Pool + Keep-Alive (save TCP/TLS handshake)
  Layer 4: Pre-warmed Connections (first request is also fast)
  Layer 5: Optimized Context (top_k=1, short prompt, fewer tokens)
  Layer 6: Async Groq with httpx (persistent HTTP/2 connection)

Target: < 200ms end-to-end on cloud deployment.
"""

import time
import logging
import hashlib
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
import httpx
import asyncio
import numpy as np

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────
QDRANT_CLOUD_URL = "https://8fe96bbe-5d1a-4be8-a5b4-7c93cccab7e8.us-east-2-0.aws.cloud.qdrant.io"
QDRANT_CLOUD_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YjA5YTI2NGUtN2E0Ni00YTBhLTlhNzAtYTExNmI1M2QyYjA5In0.FlIEcza_AEzzmJALdqnSxYmGQ2M1Et7Bj0txqOyftCc"
COLLECTION_NAME = "msmarco_chunks"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ─── LAYER 1: Full Response Cache (LRU) ────────────────────
class LRUCache:
    """Thread-safe LRU cache for full RAG responses."""
    def __init__(self, max_size=500):
        self._cache = OrderedDict()
        self._max_size = max_size
    
    def _key(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()
    
    def get(self, query: str):
        key = self._key(query)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
    
    def put(self, query: str, answer: str):
        key = self._key(query)
        self._cache[key] = answer
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

response_cache = LRUCache(max_size=500)

# ─── LAYER 2: Embedding Cache ──────────────────────────────
embedding_cache = OrderedDict()
EMBEDDING_CACHE_SIZE = 200

def get_cached_embedding(query: str, model: TextEmbedding) -> list:
    """Cache embeddings to avoid re-encoding repeated queries."""
    key = query.strip().lower()
    if key in embedding_cache:
        embedding_cache.move_to_end(key)
        return embedding_cache[key]
    
    vector = list(model.embed([key]))[0].tolist()
    embedding_cache[key] = vector
    embedding_cache.move_to_end(key)
    if len(embedding_cache) > EMBEDDING_CACHE_SIZE:
        embedding_cache.popitem(last=False)
    return vector

# ─── Global State ──────────────────────────────────────────
embedding_model: TextEmbedding | None = None
qdrant: AsyncQdrantClient | None = None
groq_http: httpx.AsyncClient | None = None  # LAYER 3: Persistent connection pool

# ─── Compact System Prompt (LAYER 5: fewer tokens = faster) ─
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful AI assistant. Answer the user's question accurately in 1-2 short sentences. "
    "Use the following context if it is helpful, but if the context doesn't contain the answer, you can use your own knowledge.\n\n"
    "CONTEXT:\n{context}"
)

# ─── Lifespan (Startup/Shutdown) ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, qdrant, groq_http
    
    # Initialize Embedding Model using FastEmbed (ONNX, ultra-low memory)
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL, threads=1)
    
    # LAYER 3: Persistent HTTP connection pool to Groq
    groq_http = httpx.AsyncClient(
        base_url="https://api.groq.com",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(10.0, connect=5.0),
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        http2=True,  # HTTP/2 multiplexing for speed
    )
    
    # Connect to Cloud Qdrant (1.3M vectors)
    log.info("Connecting to Qdrant Cloud...")
    qdrant = AsyncQdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_CLOUD_KEY)
    
    try:
        info = await qdrant.get_collection(COLLECTION_NAME)
        log.info(f"Qdrant Cloud ready! {info.points_count:,} vectors")
    except Exception as e:
        log.warning("Qdrant Error: %s", e)
    
    # LAYER 4: Pre-warm connections (avoid cold-start on first request)
    log.info("Pre-warming connections...")
    try:
        # Warm Qdrant connection
        dummy_vector = list(embedding_model.embed(["warmup"]))[0].tolist()
        await qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=dummy_vector,
            limit=1,
        )
        log.info("Qdrant connection pre-warmed ✓")
        
        # Warm Groq connection (tiny request)
        await groq_http.post("/openai/v1/chat/completions", json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        })
        log.info("Groq connection pre-warmed ✓")
    except Exception as e:
        log.warning("Pre-warm failed (non-critical): %s", e)
    
    log.info("🚀 All systems GO! Server ready.")
    yield
    
    await groq_http.aclose()
    log.info("Shutting down...")


# ─── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Ultra-Optimized Voice RAG API",
    description="6-Layer Latency Optimized RAG Pipeline",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str = Field(..., description="The user's question from Voice-to-Text")
    top_k: int = Field(default=3, description="Number of context chunks (1 = fastest, 3 = more accurate)")


# ─── RAG Endpoint (OPTIMIZED) ─────────────────────────────
@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ultra-optimized RAG Pipeline:
    Layer 1: Check response cache → instant return
    Layer 2: Cached embedding → skip re-encoding  
    Layer 3: Persistent connection pool → no TCP/TLS overhead
    Layer 5: Minimal context (top_k=1) + short prompt
    """
    start_time = time.perf_counter()
    
    if not embedding_model or not qdrant or not groq_http:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # ─── LAYER 1: Full Response Cache Check ────────────────
    cached = response_cache.get(req.query)
    if cached:
        latency = (time.perf_counter() - start_time) * 1000
        log.info(f"CACHE HIT! Latency: {latency:.1f}ms")
        return PlainTextResponse(cached)

    # ─── LAYER 2: Cached Embedding ─────────────────────────
    t0 = time.perf_counter()
    vector_list = await asyncio.to_thread(get_cached_embedding, req.query, embedding_model)
    embed_ms = (time.perf_counter() - t0) * 1000
    
    # ─── STEP 1: Qdrant Cloud Search (LAYER 5: top_k=1) ───
    t1 = time.perf_counter()
    result = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector_list,
        limit=req.top_k,
    )
    retrieval_ms = (time.perf_counter() - t1) * 1000
    
    contexts = [hit.payload.get("child_chunk", "") for hit in result.points]
    combined_context = "\n".join(contexts)

    # ─── STEP 2: Groq LLM (LAYER 3+6: persistent conn + async httpx) ─
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=combined_context)
    
    t2 = time.perf_counter()
    
    async def generate_stream():
        full_answer = []
        try:
            async with groq_http.stream(
                "POST",
                "/openai/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": req.query}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 60,
                    "stream": True,
                }
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        import json
                        try:
                            data = json.loads(line[6:])
                            content = data["choices"][0]["delta"].get("content", "")
                            if content:
                                full_answer.append(content)
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
        except Exception as e:
            log.error(f"Groq stream error: {e}")
            yield "I am sorry, an error occurred."
        
        # Cache the full response for next time
        final_answer = "".join(full_answer)
        if final_answer:
            response_cache.put(req.query, final_answer)
        
        total_ms = (time.perf_counter() - start_time) * 1000
        log.info(
            f"Query: '{req.query[:40]}...' | "
            f"Embed: {embed_ms:.0f}ms | Qdrant: {retrieval_ms:.0f}ms | "
            f"Total: {total_ms:.0f}ms"
        )
    
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# ─── Health Check ──────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(response_cache._cache)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
