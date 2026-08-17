"""
ElevenLabs Realtime STT + Speculative Retrieval Pipeline Module.
Extracted from: https://github.com/AdityaGupta14-creator/Elevenlabs-integration/blob/main/stt_test.py

This module provides the core classes for:
- Member2RetrievalClient: Calls the retrieval API endpoint
- ContextRingBuffer: Fixed-size ring buffer for speculative result caching
- LatencyTracker: Records end-to-end latency metrics with P50/P70/P100
- SpeculativeSearchManager: Coordinates speculative search triggers and verification

Optimized: HTTP/2 connection pooling, reduced timeouts, aggressive speculative triggers.
"""

import asyncio
import time
import uuid
from collections import deque
from typing import Dict, List, Optional, Tuple

import httpx

# Member 2 Retrieval API Endpoint
RETRIEVAL_API_URL = "https://goa2.onrender.com/ask"


class Member2RetrievalClient:
    """
    Client for Member 2's Retrieval API (https://goa2.onrender.com/ask).
    
    Optimized: HTTP/2 enabled, connection pooling, reduced timeout for fast failure.
    """

    def __init__(self, endpoint_url: str = RETRIEVAL_API_URL):
        self.endpoint_url = endpoint_url
        self.client = httpx.AsyncClient(
            timeout=8.0,  # Reduced from 15s — fail fast on slow responses
            http2=True,   # HTTP/2 multiplexing for connection reuse
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
        )

    async def retrieve(self, query: str) -> str:
        """
        Sends query to Member 2 retrieval API and returns context results.
        """
        try:
            response = await self.client.post(
                self.endpoint_url,
                json={"query": query},
            )
            if response.status_code == 200:
                return response.text.strip()
            else:
                return f"Error {response.status_code}: {response.text[:200]}"
        except Exception as e:
            return f"Retrieval connection error: {e}"

    async def close(self):
        await self.client.aclose()


class ContextRingBuffer:
    """
    Fixed-size ring buffer storing speculative retrieval results in memory.
    Allows instant lookup and query similarity verification when final transcript arrives.
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, query: str, result: str):
        words = set(w.lower() for w in query.strip().split() if len(w) > 2)
        self.buffer.append({
            "query": query,
            "words": words,
            "result": result,
            "timestamp": time.time(),
        })

    def find_best_match(self, final_query: str) -> Tuple[Optional[Dict], float]:
        """
        Finds the speculative result in the ring buffer most relevant to the final query.
        Returns: (best_matching_entry, similarity_score_0_to_1)
        """
        if not self.buffer:
            return None, 0.0

        final_words = set(w.lower() for w in final_query.strip().split() if len(w) > 2)
        if not final_words:
            return None, 0.0

        best_entry = None
        best_score = 0.0

        for entry in self.buffer:
            entry_words = entry["words"]
            if not entry_words:
                continue

            # Calculate Jaccard similarity score between speculative query and final query
            intersection = len(final_words.intersection(entry_words))
            union = len(final_words.union(entry_words))
            score = intersection / union if union > 0 else 0.0

            # Boost score if speculative query is an exact subset prefix of final query
            if entry_words.issubset(final_words):
                score = max(score, len(entry_words) / len(final_words))

            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry, round(best_score, 2)

    def clear(self):
        self.buffer.clear()


class LatencyTracker:
    """
    Records end-to-end latency metrics for each user query utterance.
    Calculates P50, P70, and P100 benchmarks.
    """

    def __init__(self):
        self.history: List[Dict[str, float]] = []
        self.reset_current_session()

    def reset_current_session(self):
        self.request_id = str(uuid.uuid4())[:8]
        self.audio_start: Optional[float] = None
        self.first_partial_transcript: Optional[float] = None
        self.speculative_search_start: Optional[float] = None
        self.speculative_search_end: Optional[float] = None
        self.final_transcript: Optional[float] = None
        self.final_verification: Optional[float] = None
        self.response_complete: Optional[float] = None
        self.cache_hit: bool = False

    def mark_audio_start(self):
        if self.audio_start is None:
            self.audio_start = time.time()

    def mark_first_partial(self):
        if self.first_partial_transcript is None and self.audio_start is not None:
            self.first_partial_transcript = time.time()

    def mark_speculative_start(self):
        self.speculative_search_start = time.time()

    def mark_speculative_end(self):
        self.speculative_search_end = time.time()

    def mark_final_transcript(self):
        self.final_transcript = time.time()

    def mark_final_verification(self, is_cache_hit: bool):
        self.final_verification = time.time()
        self.cache_hit = is_cache_hit

    def mark_response_complete(self):
        self.response_complete = time.time()

    def get_session_metrics(self) -> Dict:
        """Calculate and return metrics for the current session."""
        now = self.response_complete or time.time()
        ref_time = self.audio_start or now

        stt_latency = (self.first_partial_transcript - ref_time) * 1000 if self.first_partial_transcript else 0.0

        spec_latency = (
            (self.speculative_search_end - self.speculative_search_start) * 1000
            if (self.speculative_search_start and self.speculative_search_end)
            else 0.0
        )

        final_verif_latency = (
            (self.final_verification - self.final_transcript) * 1000
            if (self.final_transcript and self.final_verification)
            else 0.0
        )

        total_latency = (now - ref_time) * 1000

        metrics = {
            "request_id": self.request_id,
            "stt_partial_ms": round(stt_latency, 1),
            "speculative_retrieval_ms": round(spec_latency, 1),
            "final_verification_ms": round(final_verif_latency, 1),
            "total_ms": round(total_latency, 1),
            "cache_hit": self.cache_hit,
        }

        self.history.append(metrics)
        return metrics

    def get_percentiles(self) -> Dict:
        """Get P50, P70, P100 from history."""
        totals = sorted(m["total_ms"] for m in self.history)
        n = len(totals)
        if n == 0:
            return {"p50": 0, "p70": 0, "p100": 0}
        p50 = totals[int(0.50 * (n - 1))]
        p70 = totals[int(0.70 * (n - 1))]
        p100 = totals[-1]
        return {"p50": round(p50, 1), "p70": round(p70, 1), "p100": round(p100, 1)}


class SpeculativeSearchManager:
    """
    Coordinates partial transcript speculative triggers, ring buffer context storage,
    and final query verification with Member 2's Retrieval API.
    
    Optimized: lower trigger threshold (3 words, 2-word gap) for earlier speculative search.
    """

    def __init__(self, retrieval_client: Member2RetrievalClient, tracker: LatencyTracker):
        self.client = retrieval_client
        self.tracker = tracker
        self.ring_buffer = ContextRingBuffer(capacity=10)
        self.active_task: Optional[asyncio.Task] = None
        self.last_triggered_word_count: int = 0
        self.search_counter: int = 0

    async def _run_speculative_search(self, search_id: int, query: str):
        try:
            print(f"\n---> [SPECULATIVE SEARCH #{search_id}] Triggered for: '{query}'")
            self.tracker.mark_speculative_start()
            result = await self.client.retrieve(query)
            self.tracker.mark_speculative_end()

            # Store result in context ring buffer
            self.ring_buffer.add(query, result)
            print(f"\n<--- [SPECULATIVE SEARCH #{search_id} COMPLETED] Stored in Ring Buffer ({len(result)} chars)")
        except asyncio.CancelledError:
            print(f"\n[!] [SPECULATIVE SEARCH #{search_id} CANCELLED] Cancelled stale search.")
            raise

    def handle_partial(self, partial_text: str):
        self.tracker.mark_first_partial()
        words = [w for w in partial_text.strip().split() if w]
        num_words = len(words)

        # Optimized: Trigger speculative search at >= 3 words with 2-word gap
        # (was: >= 4 words with 3-word gap) — starts retrieval sooner while user speaks
        if num_words >= 3:
            if self.last_triggered_word_count == 0 or (num_words - self.last_triggered_word_count) >= 2:
                self.last_triggered_word_count = num_words
                self.search_counter += 1
                search_id = self.search_counter

                # Cancel stale in-flight search task
                if self.active_task and not self.active_task.done():
                    self.active_task.cancel()

                # Launch new speculative search task
                self.active_task = asyncio.create_task(
                    self._run_speculative_search(search_id, partial_text.strip())
                )

    async def handle_final(self, final_text: str) -> Tuple[str, bool, float]:
        """
        Handle final transcript: verify against ring buffer or do fresh retrieval.
        Returns: (retrieved_context, is_cache_hit, similarity_score)
        """
        self.tracker.mark_final_transcript()

        # Cancel any remaining speculative search when final committed transcript arrives
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()

        self.last_triggered_word_count = 0
        final_query = final_text.strip()

        print(f"\n[VERIFICATION] Comparing final query '{final_query}' against Ring Buffer...")

        # Final-query verification against Ring Buffer
        best_match, score = self.ring_buffer.find_best_match(final_query)

        if best_match and score >= 0.65:
            # GOOD MATCH: Use speculative result instantly from memory
            self.tracker.mark_final_verification(is_cache_hit=True)
            retrieved_context = best_match["result"]
            print(f"  [+] CACHE MATCH (Score: {score}) -> Using buffered speculative result for: '{best_match['query']}'")
            return retrieved_context, True, score
        else:
            # POOR MATCH / MISS: Execute fresh retrieval
            print(f"  [-] CACHE MISS (Score: {score}) -> Executing fresh final retrieval...")
            self.tracker.mark_speculative_start()
            retrieved_context = await self.client.retrieve(final_query)
            self.tracker.mark_speculative_end()
            self.tracker.mark_final_verification(is_cache_hit=False)
            return retrieved_context, False, score

    def reset(self):
        """Reset for next utterance."""
        self.last_triggered_word_count = 0
        self.search_counter = 0
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
        self.active_task = None
        self.tracker.reset_current_session()
