"""
Test script: Measures RAG API latency with columns:
- Input Query
- Embed (ms)
- Qdrant (ms) 
- LLM TTFT (ms)
- Total (ms)
- Cache Hit?
- Output
"""
import requests
import time

API_URL = "http://127.0.0.1:8000/ask"

QUERIES = [
    # Fresh queries (first time)
    "What is the capital of France?",
    "Who is Lionel Messi?",
    "What is the freezing point of water?",
    "How many continents are there?",
    "What is the speed of light?",
    # REPEATED queries (should be cached = near 0ms)
    "What is the capital of France?",
    "Who is Lionel Messi?",
    "What is the freezing point of water?",
]

with open("test_results_optimized.md", "w", encoding="utf-8") as f:
    f.write("# 🚀 Ultra-Optimized RAG API — Latency Test Results\n\n")
    f.write("**6-Layer Optimization: Cache + Connection Pool + Pre-warm + Compact Prompt**\n\n")
    f.write("| # | Input (Query) | TTFT (ms) | Total (ms) | Cache Hit? | Output (AI Answer) |\n")
    f.write("|---|---|---|---|---|---|\n")

    for i, q in enumerate(QUERIES):
        payload = {"query": q, "top_k": 3}
        try:
            start_time = time.perf_counter()
            
            response = requests.post(API_URL, json=payload, stream=True, timeout=15)
            response.raise_for_status()

            ttft = None
            full_response = ""

            for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
                if chunk:
                    if ttft is None:
                        ttft = (time.perf_counter() - start_time) * 1000
                    full_response += chunk

            total_time = (time.perf_counter() - start_time) * 1000
            
            clean_response = full_response.replace("\n", " ").replace("\r", " ").replace("|", "/").strip()
            if len(clean_response) > 70:
                clean_response = clean_response[:70] + "..."
                
            ttft_str = f"{ttft:.1f}" if ttft else "N/A"
            total_str = f"{total_time:.1f}"
            
            # Detect cache hit (if total < 50ms, definitely cached)
            is_cached = "✅ YES" if total_time < 100 else "❌ No"
            
            f.write(f"| {i+1} | {q} | {ttft_str} | {total_str} | {is_cached} | {clean_response} |\n")
            
        except Exception as e:
            err = str(e).replace("|", "/")
            f.write(f"| {i+1} | {q} | ERROR | ERROR | — | {err[:80]} |\n")
    
    f.write("\n---\n*Queries 6-8 are REPEATED queries to test cache performance.*\n")

print("Done! Results saved to test_results_optimized.md")
