"""
Speculative Voice-RAG: ULTRA-OPTIMIZED RAG API
================================================
Deployed on Vercel Serverless. Uses a module-level
requests.Session to reuse connections and avoid
file descriptor exhaustion (Errno 16).
"""

import time
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import json 
import logging
import traceback
import hashlib
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
import requests

# ─── Global Session (reuse TCP connections, avoid FD exhaustion) ───
# Created at MODULE LEVEL — shared across all invocations in the same container
_session = requests.Session()

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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}/pipeline/feature-extraction"


# ─── LAYER 1: Full Response Cache (LRU) ────────────────────
class LRUCache:
    def __init__(self, max_size=500):
        self._cache = OrderedDict()
        self._max_size = max_size

    def _key(self, query: str, mode: str) -> str:
        key_str = f"{query.strip().lower()}_{mode}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, mode: str):
        key = self._key(query, mode)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, query: str, mode: str, answer: str):
        key = self._key(query, mode)
        self._cache[key] = answer
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


response_cache = LRUCache(max_size=500)

# ─── LAYER 2: Embedding Cache ──────────────────────────────
embedding_cache = OrderedDict()
EMBEDDING_CACHE_SIZE = 200


def get_embedding_sync(query: str) -> list:
    """Get embedding via HF API using module-level session."""
    key = query.strip().lower()
    if key in embedding_cache:
        embedding_cache.move_to_end(key)
        return embedding_cache[key]

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    else:
        return [0.0] * 384

    resp = _session.post(HF_API_URL, json={"inputs": [key]}, headers=headers, timeout=15.0)
    if resp.status_code != 200:
        raise Exception(f"HF API Error ({resp.status_code}): {resp.text}")

    vector = resp.json()[0]

    embedding_cache[key] = vector
    embedding_cache.move_to_end(key)
    if len(embedding_cache) > EMBEDDING_CACHE_SIZE:
        embedding_cache.popitem(last=False)
    return vector


# ─── Global State ──────────────────────────────────────────
qdrant: AsyncQdrantClient | None = None

# ─── Compact System Prompt ─────────────────────────────────
SYSTEM_PROMPT_STRICT = (
    "You are a highly intelligent and advanced Voice Assistant. Follow these rules STRICTLY:\n"
    "1. DEFAULT TO ENGLISH: You must reply in English by default. ONLY reply in another language if the user's input is undeniably and entirely in that other language. Do NOT announce what language you are using.\n"
    "2. CONCISE CONVERSATIONAL ANSWERS: Since your output is spoken by a Voice AI, keep your answers SHORT, natural, and to the point (1-3 sentences maximum). DO NOT give long essays or large text blocks unless the user explicitly asks for a detailed explanation.\n"
    "3. NO HINGLISH & NO FORMATTING: Never mix languages. Never use markdown formatting (no asterisks, hashes, bullet points). Use plain conversational text only.\n"
    "4. STRICT KNOWLEDGE: You MUST ONLY answer based on the provided CONTEXT. If the context does not contain the answer, you MUST decline to answer gracefully and state a clear reason (e.g., 'I do not have this information in my database' or 'My knowledge is restricted on this topic'). Do NOT use your general knowledge under any circumstances.\n\n"
    "CONTEXT:\n{context}"
)

SYSTEM_PROMPT_FULL = (
    "You are a highly intelligent and advanced Voice Assistant. Follow these rules STRICTLY:\n"
    "1. DEFAULT TO ENGLISH: You must reply in English by default. ONLY reply in another language if the user's input is undeniably and entirely in that other language. Do NOT announce what language you are using.\n"
    "2. CONCISE CONVERSATIONAL ANSWERS: Since your output is spoken by a Voice AI, keep your answers SHORT, natural, and to the point (1-3 sentences maximum). DO NOT give long essays or large text blocks unless the user explicitly asks for a detailed explanation.\n"
    "3. NO HINGLISH & NO FORMATTING: Never mix languages. Never use markdown formatting (no asterisks, hashes, bullet points). Use plain conversational text only.\n"
    "4. HYBRID KNOWLEDGE: Use the provided context to answer. If the context doesn't contain the answer, you are free to use your advanced general intelligence to provide an accurate, concise response.\n\n"
    "CONTEXT:\n{context}"
)

# ─── Lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant
    log.info("Connecting to Qdrant Cloud...")
    qdrant = AsyncQdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_CLOUD_KEY)
    try:
        info = await qdrant.get_collection(COLLECTION_NAME)
        log.info(f"Qdrant Cloud ready! {info.points_count:,} vectors")
    except Exception as e:
        log.warning("Qdrant Error: %s", e)
    log.info("Server ready.")
    yield
    log.info("Shutting down...")


# ─── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Voice RAG API",
    description="Optimized RAG Pipeline",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    top_k: int = Field(default=3, description="Number of context chunks")
    mode: str = Field(default="strict", description="Mode: 'strict' (dataset only) or 'full' (LLM knowledge fallback)")


@app.post("/ask")
async def ask(req: AskRequest):
    start_time = time.perf_counter()

    # Lazy-init Qdrant if lifespan didn't run (Vercel serverless)
    global qdrant
    if not qdrant:
        qdrant = AsyncQdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_CLOUD_KEY)

    try:
        # LAYER 1: Cache check
        cached = response_cache.get(req.query, req.mode)
        if cached:
            latency = (time.perf_counter() - start_time) * 1000
            log.info(f"CACHE HIT! Latency: {latency:.1f}ms")
            
            def cache_stream():
                yield f"data: {json.dumps({'answer': cached})}\n\n"
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(cache_stream(), media_type="text/event-stream")

        # LAYER 2: Embedding
        t0 = time.perf_counter()
        vector_list = get_embedding_sync(req.query)
        embed_ms = (time.perf_counter() - t0) * 1000

        # Qdrant search
        t1 = time.perf_counter()
        result = await qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector_list,
            limit=req.top_k,
        )
        retrieval_ms = (time.perf_counter() - t1) * 1000

        contexts = [hit.payload.get("child_chunk", "") for hit in result.points]
        combined_context = "\n".join(contexts)
    except Exception as e:
        return PlainTextResponse(f"Error: {e}\n\nTraceback: {traceback.format_exc()}", status_code=500)

    # Groq LLM streaming
    prompt_template = SYSTEM_PROMPT_FULL if req.mode == "full" else SYSTEM_PROMPT_STRICT
    system_prompt = prompt_template.format(context=combined_context)

    def generate_stream():
        full_answer = []
        try:
            groq_resp = _session.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": req.query}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "stream": True,
                },
                timeout=30.0,
                stream=True,
            )
            if groq_resp.status_code != 200:
                raise Exception(f"Groq API Error {groq_resp.status_code}: {groq_resp.text}")
            for raw_line in groq_resp.iter_lines():
                if raw_line:
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            if "error" in data:
                                yield f"data: {json.dumps({'error': str(data['error'])})}\n\n"
                            
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_answer.append(content)
                                    yield f"data: {json.dumps({'answer': content})}\n\n"
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            yield f"data: {json.dumps({'error': f'Parse error: {str(e)} on line: {line}'})}\n\n"
            groq_resp.close()
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

        final_answer = "".join(full_answer)
        if final_answer:
            response_cache.put(req.query, req.mode, final_answer)

        total_ms = (time.perf_counter() - start_time) * 1000
        log.info(
            f"Query: '{req.query[:40]}...' | "
            f"Embed: {embed_ms:.0f}ms | Qdrant: {retrieval_ms:.0f}ms | "
            f"Total: {total_ms:.0f}ms"
        )

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(response_cache._cache)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
