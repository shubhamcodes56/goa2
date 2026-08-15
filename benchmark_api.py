"""
Speculative Voice-RAG: Search Benchmark API (Member 2)
=======================================================
FastAPI server that accepts text queries, encodes them using
a multilingual SentenceTransformer, and searches the Qdrant
vector database for the most relevant Hindi/English passages.

Designed for < 200ms search latency (Hackathon requirement).
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────
QDRANT_URL = "https://8fe96bbe-5d1a-4be8-a5b4-7c93cccab7e8.us-east-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6YjA5YTI2NGUtN2E0Ni00YTBhLTlhNzAtYTExNmI1M2QyYjA5In0.FlIEcza_AEzzmJALdqnSxYmGQ2M1Et7Bj0txqOyftCc"
COLLECTION_NAME = "msmarco_chunks"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ─── Global State ──────────────────────────────────────────
model: SentenceTransformer | None = None
qdrant: QdrantClient | None = None


# ─── Lifespan (modern FastAPI startup/shutdown) ────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect to Qdrant Cloud on startup."""
    global model, qdrant
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Connecting to Qdrant Cloud at %s", QDRANT_URL)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        collection_info = qdrant.get_collection(COLLECTION_NAME)
        log.info(
            "Qdrant ready! Collection '%s' has %s vectors",
            COLLECTION_NAME,
            f"{collection_info.points_count:,}",
        )
    except Exception as e:
        log.warning("Collection not found yet. Did you run the Kaggle script? Error: %s", e)

    yield  # App is running
    log.info("Shutting down...")


# ─── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Speculative Voice-RAG Search API",
    description="Cross-lingual vector search for Hindi & English passages",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Member 1 and Member 3 frontends to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., description="The user's question (partial or full)")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of results")
    language: str | None = Field(default=None, description="Filter by 'hi' or 'en'")


class SearchResult(BaseModel):
    score: float
    child_chunk: str
    parent_context: str
    language: str
    passage_id: str


class SearchResponse(BaseModel):
    latency_ms: float
    num_results: int
    results: list[SearchResult]
    message: str = "Success"


# ─── Search Endpoint ───────────────────────────────────────
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Encode the query and search Qdrant for the best matching passages.
    Supports cross-lingual search: English query → Hindi results and vice versa.
    Supports optional language filtering.
    """
    if not model or not qdrant:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start = time.perf_counter()

    # 1. Encode query to vector
    vector = model.encode(req.query).tolist()

    # 2. Build optional language filter
    query_filter = None
    if req.language:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=req.language))]
        )

    # 3. Search Qdrant
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=req.top_k,
    ).points

    latency_ms = (time.perf_counter() - start) * 1000

    # 4. Format results
    results = [
        SearchResult(
            score=hit.score,
            child_chunk=hit.payload.get("child_chunk", ""),
            parent_context=hit.payload.get("parent_context", ""),
            language=hit.payload.get("language", "unknown"),
            passage_id=hit.payload.get("passage_id", ""),
        )
        for hit in hits
    ]

    return SearchResponse(
        latency_ms=latency_ms,
        num_results=len(results),
        results=results,
        message=f"Search completed in {latency_ms:.2f} ms",
    )


# ─── Health Check ──────────────────────────────────────────
@app.get("/health")
async def health():
    """Quick health check for monitoring."""
    if not model or not qdrant:
        return {"status": "loading"}
    return {"status": "ok", "collection": COLLECTION_NAME}


# ─── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
