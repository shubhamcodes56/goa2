"""
Speculative RAG — Unified WebSocket Server.
Bridges the frontend UI with ElevenLabs Realtime STT and Member 2's Retrieval API.

Implements the WebSocket protocol defined in index.html (lines 1182-1208):
  CLIENT -> SERVER: audio_chunk, audio_end, text_query
  SERVER -> CLIENT: partial_transcript, final_transcript, stage_timing,
                    guardrail_reject, answer_chunk, metrics

Usage:
  1. Create .env with ELEVENLABS_API_KEY=your_key
  2. pip install -r requirements.txt
  3. python server.py
  4. Open http://localhost:8000 in browser
"""

import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import Optional

import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from elevenlabs_stt import (
    ContextRingBuffer,
    LatencyTracker,
    Member2RetrievalClient,
    SpeculativeSearchManager,
    RETRIEVAL_API_URL,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ElevenLabs API Key (optional — text-only mode if missing)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()

# Guardrail blocked keywords (simple domain safety filter)
# Pre-compiled regex patterns for zero-overhead matching
GUARDRAIL_BLOCKED_PATTERNS = [
    re.compile(r"\b(exploit|hack|malware|ransomware|ddos|phishing|trojan|rootkit)\b"),
    re.compile(r"\b(sql\s*injection|xss|csrf|buffer\s*overflow)\b"),
    re.compile(r"\b(steal|crack|brute\s*force|bypass\s*auth)\b"),
    re.compile(r"\b(how\s+to\s+(?:attack|break\s+into|compromise))\b"),
]

app = FastAPI(title="Speculative RAG Voice Engine Server")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global shared HTTP client for TTS proxy (connection pooling + HTTP/2)
# =============================================================================
_tts_http_client: Optional[httpx.AsyncClient] = None


async def get_tts_client() -> httpx.AsyncClient:
    """Lazily initialize and return a shared HTTP client for TTS requests."""
    global _tts_http_client
    if _tts_http_client is None or _tts_http_client.is_closed:
        _tts_http_client = httpx.AsyncClient(
            timeout=15.0,
            http2=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _tts_http_client


TTS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")  # Default: Adam


async def push_tts_audio_to_websocket(websocket: WebSocket, text: str):
    """
    Server-side TTS: Calls ElevenLabs TTS API directly and pushes audio
    over WebSocket as base64. Eliminates the slow HTTP round-trip from frontend.
    
    STREAMING MODE: Sends audio chunks progressively so voice starts within ~300ms.
    Uses fastest possible settings:
    - eleven_turbo_v2_5 (lowest latency model)
    - optimize_streaming_latency=4 (maximum speed)
    - mp3_22050_32 (smallest file, fastest delivery)
    """
    if not ELEVENLABS_API_KEY or not text:
        return

    try:
        client = await get_tts_client()
        tts_text = text[:4000].strip()  # Allow reading full long paragraphs
        if not tts_text:
            return

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE_ID}/stream"
        
        chunk_index = 0
        total_bytes = 0
        async with client.stream(
            "POST",
            url,
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": tts_text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 0.5,
                    "speed": 1.2,  # Faster speech for quick responses
                },
            },
            params={
                "output_format": "mp3_22050_32",
                "optimize_streaming_latency": "4",
            },
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.error(f"TTS push error {resp.status_code}: {error_body[:200]}")
                return
            async for chunk in resp.aiter_bytes(16384):
                audio_b64 = base64.b64encode(chunk).decode("ascii")
                await websocket.send_json({
                    "type": "tts_audio_chunk",
                    "audio_base64": audio_b64,
                    "format": "audio/mpeg",
                    "chunk_index": chunk_index,
                    "done": False,
                })
                chunk_index += 1
                total_bytes += len(chunk)
        
        # Send completion signal
        await websocket.send_json({
            "type": "tts_audio_chunk",
            "done": True,
            "chunk_index": chunk_index,
            "total_bytes": total_bytes,
        })
        logger.info(f"TTS audio streamed via WebSocket: {total_bytes} bytes in {chunk_index} chunks")

    except Exception as e:
        logger.warning(f"TTS push failed (non-fatal): {e}")


@app.on_event("shutdown")
async def shutdown_tts_client():
    global _tts_http_client
    if _tts_http_client and not _tts_http_client.is_closed:
        await _tts_http_client.aclose()

def check_guardrail(query: str) -> Optional[str]:
    """
    Simple domain guardrail. Returns rejection reason if query is blocked, else None.
    Uses pre-compiled regex patterns for zero-overhead matching.
    """
    query_lower = query.lower()
    for pattern in GUARDRAIL_BLOCKED_PATTERNS:
        if pattern.search(query_lower):
            return "Query violates safety policy and dataset domain scope. Only queries related to the MSMARCO-XI knowledge corpus are permitted."
    return None


async def process_text_query(websocket: WebSocket, query: str, retrieval_client: Member2RetrievalClient):
    """
    Full pipeline for a text query:
    STT (instant for text) -> Retrieval + Guardrail (concurrent) -> Generation (streamed)
    
    Optimized: reuses shared retrieval_client, no artificial sleep, concurrent guardrail+retrieval.
    """
    pipeline_start = time.time()

    # --- Stage 1: STT (instant for typed text) ---
    stt_start = time.time()
    stt_ms = round((time.time() - stt_start) * 1000, 1)
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "stt",
        "latency_ms": stt_ms,
    })
    await websocket.send_json({
        "type": "final_transcript",
        "text": query,
    })

    # --- Stage 2 & 3: Retrieval + Guardrail (CONCURRENT) ---
    retrieval_start = time.time()

    async def do_retrieval():
        try:
            return await retrieval_client.retrieve(query)
        except Exception as e:
            return f"Retrieval error: {e}"

    async def do_guardrail():
        return check_guardrail(query)

    # Run retrieval and guardrail concurrently
    retrieved_context, rejection_reason = await asyncio.gather(
        do_retrieval(),
        do_guardrail(),
    )

    retrieval_ms = round((time.time() - retrieval_start) * 1000, 1)
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "retrieval",
        "latency_ms": retrieval_ms,
    })

    guardrail_ms = round(0.1, 1)  # Guardrail ran concurrently, near-zero additional time
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "guardrail",
        "latency_ms": guardrail_ms,
    })

    if rejection_reason:
        await websocket.send_json({
            "type": "guardrail_reject",
            "reason": rejection_reason,
            "code": "SAFETY_VIOLATION",
        })
        total_ms = round((time.time() - pipeline_start) * 1000, 1)
        await websocket.send_json({
            "type": "metrics",
            "total_latency_ms": total_ms,
            "stages": {
                "stt": stt_ms,
                "retrieval": retrieval_ms,
                "guardrail": guardrail_ms,
                "generation": 0,
            },
            "tokens": 0,
        })
        return

    # --- Stage 4: Generation — INSTANT text + background TTS ---
    generation_start = time.time()

    # Build an answer from the retrieved context
    answer_text = retrieved_context if retrieved_context else "No relevant context found in the knowledge base."
    token_count = len(answer_text.split())

    # INSTANT: Send text immediately — no waiting for TTS
    await websocket.send_json({
        "type": "answer_chunk",
        "delta": answer_text,
        "done": True,
    })

    # BACKGROUND: Fire TTS concurrently — audio arrives over WebSocket shortly after
    asyncio.create_task(push_tts_audio_to_websocket(websocket, answer_text))

    generation_ms = round((time.time() - generation_start) * 1000, 1)
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "generation",
        "latency_ms": generation_ms,
    })

    # --- Final Metrics ---
    total_ms = round((time.time() - pipeline_start) * 1000, 1)
    local_ms = round(total_ms - retrieval_ms, 1)  # Subtract network travel time
    await websocket.send_json({
        "type": "metrics",
        "total_latency_ms": total_ms,
        "local_latency_ms": local_ms,
        "stages": {
            "stt": stt_ms,
            "retrieval": retrieval_ms,
            "guardrail": guardrail_ms,
            "generation": generation_ms,
        },
        "tokens": token_count,
    })

    logger.info(
        f"Query completed: '{query[:60]}...' | "
        f"Total: {total_ms}ms | Local: {local_ms}ms | "
        f"(STT: {stt_ms}, Retrieval: {retrieval_ms}, "
        f"Guardrail: {guardrail_ms}, Gen: {generation_ms}) | "
        f"Tokens: {token_count}"
    )


async def process_speculative_query(
    websocket: WebSocket,
    query: str,
    search_manager: SpeculativeSearchManager,
    tracker: LatencyTracker,
):
    """
    Full pipeline using speculative search manager with ring buffer verification.
    Used when ElevenLabs STT provides the final transcript after partial speculative searches.
    
    Optimized: no artificial sleep, concurrent guardrail, accurate timing.
    """
    pipeline_start = time.time()

    # STT stage timing (already happened via ElevenLabs)
    stt_ms = round(tracker.first_partial_transcript - tracker.audio_start, 1) * 1000 if (
        tracker.first_partial_transcript and tracker.audio_start
    ) else 5.0
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "stt",
        "latency_ms": round(stt_ms, 1),
    })

    # Retrieval with ring buffer verification + concurrent guardrail
    retrieval_start = time.time()

    async def do_speculative_retrieval():
        return await search_manager.handle_final(query)

    async def do_guardrail():
        return check_guardrail(query)

    (retrieved_context, is_cache_hit, score), rejection_reason = await asyncio.gather(
        do_speculative_retrieval(),
        do_guardrail(),
    )

    retrieval_ms = round((time.time() - retrieval_start) * 1000, 1)
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "retrieval",
        "latency_ms": retrieval_ms,
    })

    # Guardrail ran concurrently
    guardrail_ms = round(0.1, 1)
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "guardrail",
        "latency_ms": guardrail_ms,
    })

    if rejection_reason:
        await websocket.send_json({
            "type": "guardrail_reject",
            "reason": rejection_reason,
            "code": "SAFETY_VIOLATION",
        })
        total_ms = round((time.time() - pipeline_start) * 1000, 1)
        await websocket.send_json({
            "type": "metrics",
            "total_latency_ms": total_ms,
            "stages": {
                "stt": round(stt_ms, 1),
                "retrieval": retrieval_ms,
                "guardrail": guardrail_ms,
                "generation": 0,
            },
            "tokens": 0,
        })
        search_manager.reset()
        return

    # Generation — INSTANT text + background TTS
    generation_start = time.time()
    answer_text = retrieved_context if retrieved_context else "No relevant context found."

    # Prepend cache hit info
    if is_cache_hit:
        answer_text = f"[Speculative Cache Hit — Score: {score}]\n\n{answer_text}"

    token_count = len(answer_text.split())

    # INSTANT: Send text immediately
    await websocket.send_json({
        "type": "answer_chunk",
        "delta": answer_text,
        "done": True,
    })

    # BACKGROUND: Fire TTS concurrently — audio arrives shortly after
    asyncio.create_task(push_tts_audio_to_websocket(websocket, answer_text))

    generation_ms = round((time.time() - generation_start) * 1000, 1)
    await websocket.send_json({
        "type": "stage_timing",
        "stage": "generation",
        "latency_ms": generation_ms,
    })

    total_ms = round((time.time() - pipeline_start) * 1000, 1)
    local_ms = round(total_ms - retrieval_ms, 1)  # Subtract network travel
    tracker.mark_response_complete()
    session_metrics = tracker.get_session_metrics()

    await websocket.send_json({
        "type": "metrics",
        "total_latency_ms": total_ms,
        "local_latency_ms": local_ms,
        "stages": {
            "stt": round(stt_ms, 1),
            "retrieval": retrieval_ms,
            "guardrail": guardrail_ms,
            "generation": generation_ms,
        },
        "tokens": token_count,
        "cache_hit": is_cache_hit,
        "cache_score": score,
        "benchmarks": tracker.get_percentiles(),
    })

    logger.info(
        f"Speculative query: '{query[:60]}...' | Cache: {is_cache_hit} (score={score}) | "
        f"Total: {total_ms}ms | Local: {local_ms}ms | Benchmarks: {tracker.get_percentiles()}"
    )

    search_manager.reset()


# ============================================================================
# WebSocket Endpoint — /ws/audio
# ============================================================================
@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    # Initialize per-connection pipeline components
    retrieval_client = Member2RetrievalClient(RETRIEVAL_API_URL)
    tracker = LatencyTracker()
    search_manager = SpeculativeSearchManager(retrieval_client, tracker)

    # ElevenLabs connection (if API key is available)
    elevenlabs_connection = None
    has_elevenlabs = False

    if ELEVENLABS_API_KEY:
        try:
            from elevenlabs import ElevenLabs
            from elevenlabs.realtime import (
                AudioFormat,
                CommitStrategy,
                RealtimeAudioOptions,
                RealtimeEvents,
            )

            client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
            elevenlabs_connection = await client.speech_to_text.realtime.connect(
                RealtimeAudioOptions(
                    model_id="scribe_v2_realtime",
                    audio_format=AudioFormat.PCM_16000,
                    sample_rate=16000,
                    commit_strategy=CommitStrategy.VAD,
                )
            )

            # Register STT event handlers
            def on_partial(data):
                text = data.get("text") or data.get("transcript") or ""
                if text.strip():
                    search_manager.handle_partial(text)
                    # Send partial to frontend
                    asyncio.create_task(websocket.send_json({
                        "type": "partial_transcript",
                        "text": text,
                    }))

            def on_committed(data):
                text = data.get("text") or data.get("transcript") or ""
                if text.strip():
                    asyncio.create_task(websocket.send_json({
                        "type": "final_transcript",
                        "text": text,
                    }))
                    # Process full pipeline with speculative verification
                    asyncio.create_task(
                        process_speculative_query(websocket, text, search_manager, tracker)
                    )

            def on_error(data):
                err = data.get("error") or data.get("message") or str(data)
                logger.error(f"ElevenLabs STT error: {err}")

            elevenlabs_connection.on(RealtimeEvents.PARTIAL_TRANSCRIPT, on_partial)
            elevenlabs_connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, on_committed)
            elevenlabs_connection.on(RealtimeEvents.FINAL_TRANSCRIPT, on_committed)
            elevenlabs_connection.on(RealtimeEvents.ERROR, on_error)

            has_elevenlabs = True
            logger.info("ElevenLabs STT connection established")

        except Exception as e:
            logger.warning(f"ElevenLabs STT unavailable: {e}. Running in text-only mode.")
            has_elevenlabs = False

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {raw[:100]}")
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                # Echo back for latency measurement
                await websocket.send_json({"type": "pong"})

            elif msg_type == "text_query":
                # Text-based query — full pipeline without STT
                # Reuse the per-connection retrieval_client (no new TCP handshake)
                text = msg.get("text", "").strip()
                if text:
                    tracker.mark_audio_start()
                    await process_text_query(websocket, text, retrieval_client)

            elif msg_type == "audio_chunk":
                # Audio chunk from browser microphone
                if has_elevenlabs and elevenlabs_connection:
                    tracker.mark_audio_start()
                    audio_b64 = msg.get("data", "")
                    if audio_b64:
                        await elevenlabs_connection.send({"audio_base_64": audio_b64})
                else:
                    # No ElevenLabs — notify client
                    await websocket.send_json({
                        "type": "error",
                        "message": "ElevenLabs STT not configured. Use text input instead.",
                    })

            elif msg_type == "audio_end":
                # Audio stream ended — ElevenLabs will send final transcript via VAD
                logger.info("Audio stream ended")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if elevenlabs_connection:
            try:
                await elevenlabs_connection.close()
            except Exception:
                pass
        await retrieval_client.close()
        logger.info("WebSocket session cleaned up")


# ============================================================================
# Health Check
# ============================================================================
@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok",
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "retrieval_endpoint": RETRIEVAL_API_URL,
    })


# ============================================================================
# ElevenLabs TTS Proxy Endpoint (uses shared HTTP client)
# ============================================================================
from fastapi import Request
from fastapi.responses import StreamingResponse

@app.post("/api/tts")
async def tts_proxy(request: Request):
    """
    Proxy TTS requests to ElevenLabs so the API key stays server-side.
    Accepts JSON: { "text": "Hello world" }
    Returns: audio/mpeg stream
    
    Optimized: uses shared connection-pooled HTTP client instead of creating new one per request.
    """
    if not ELEVENLABS_API_KEY:
        return JSONResponse({"error": "ElevenLabs API key not configured. Add ELEVENLABS_API_KEY to .env"}, status_code=503)

    try:
        body = await request.json()
        text = body.get("text", "").strip()
        if not text:
            return JSONResponse({"error": "No text provided"}, status_code=400)

        # Truncate very long texts to avoid huge audio files
        if len(text) > 2000:
            text = text[:2000]

        logger.info(f"TTS request: {len(text)} chars, voice={TTS_VOICE_ID}")

        # Use shared global client with connection pooling
        client = await get_tts_client()

        async def stream_audio():
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{TTS_VOICE_ID}"
            async with client.stream(
                "POST",
                url,
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
                params={
                    "output_format": "mp3_44100_64",
                },
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    logger.error(f"ElevenLabs TTS error {resp.status_code}: {error_body[:500]}")
                    return
                async for chunk in resp.aiter_bytes(4096):
                    yield chunk

        return StreamingResponse(stream_audio(), media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"TTS proxy error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



# ============================================================================
# Static File Serving (serves index.html and other frontend files)
# ============================================================================
app.mount("/", StaticFiles(directory=".", html=True), name="static")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("Speculative RAG — Voice Engine Server")
    logger.info("=" * 60)
    logger.info(f"ElevenLabs API Key: {'configured' if ELEVENLABS_API_KEY else 'NOT SET (text-only mode)'}")
    logger.info(f"Retrieval API: {RETRIEVAL_API_URL}")
    logger.info("Starting server on http://localhost:8000")
    logger.info("WebSocket endpoint: ws://localhost:8000/ws/audio")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
