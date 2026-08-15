"""
Speculative Voice-RAG: Multi-Query Test Suite (Member 2)
=========================================================
Tests the Search API with multiple queries in both Hindi and English,
prints results in a clean table, and saves detailed JSON output.
"""

import requests
import json
import time

API_URL = "http://localhost:8000/search"

# Test queries: mix of English, Hindi, and cross-lingual
TEST_QUERIES = [
    {"query": "temperature and weather", "top_k": 3},
    {"query": "Who is the president of India?", "top_k": 3},
    {"query": "भारत की राजधानी क्या है?", "top_k": 3},
    {"query": "Amsterdam", "top_k": 2},
    {"query": "solar system planets", "top_k": 3},
]


def run_tests():
    print("=" * 70)
    print("  SPECULATIVE VOICE-RAG: SEARCH API BENCHMARK TEST")
    print("=" * 70)

    all_results = []

    for i, payload in enumerate(TEST_QUERIES, 1):
        try:
            start = time.time()
            response = requests.post(API_URL, json=payload, timeout=30)
            total_time = (time.time() - start) * 1000

            if response.status_code == 200:
                data = response.json()
                server_latency = data["latency_ms"]
                num_results = data["num_results"]

                print(f"\n{'─' * 70}")
                print(f"  Test {i}: \"{payload['query']}\"")
                print(f"{'─' * 70}")
                print(f"  Server Latency : {server_latency:.2f} ms")
                print(f"  Total Time     : {total_time:.2f} ms")
                print(f"  Results Found  : {num_results}")
                print()

                for j, res in enumerate(data["results"], 1):
                    lang_label = "HINDI" if res["language"] == "hi" else "ENGLISH"
                    chunk_preview = res["child_chunk"][:120] + "..." if len(res["child_chunk"]) > 120 else res["child_chunk"]
                    print(f"    #{j} [{lang_label}] (Score: {res['score']:.4f})")
                    print(f"       {chunk_preview}")
                    print()

                # Record result
                all_results.append({
                    "query": payload["query"],
                    "server_latency_ms": server_latency,
                    "total_request_ms": total_time,
                    "num_results": num_results,
                    "top_result_language": data["results"][0]["language"] if data["results"] else "none",
                    "top_result_score": data["results"][0]["score"] if data["results"] else 0,
                    "status": "PASS" if server_latency < 200 else "SLOW",
                })
            else:
                print(f"\n  Test {i}: FAILED (HTTP {response.status_code})")
                all_results.append({
                    "query": payload["query"],
                    "status": "ERROR",
                    "error": response.text,
                })

        except Exception as e:
            print(f"\n  Test {i}: CONNECTION FAILED - {e}")
            all_results.append({
                "query": payload["query"],
                "status": "CONNECTION_FAILED",
                "error": str(e),
            })

    # Summary
    passed = sum(1 for r in all_results if r.get("status") == "PASS")
    avg_latency = sum(r.get("server_latency_ms", 0) for r in all_results if "server_latency_ms" in r)
    latency_count = sum(1 for r in all_results if "server_latency_ms" in r)

    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Tests Passed (< 200ms) : {passed}/{len(all_results)}")
    if latency_count > 0:
        print(f"  Average Server Latency : {avg_latency / latency_count:.2f} ms")
    print(f"{'=' * 70}")

    # Save JSON
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Detailed results saved to: test_results.json")


if __name__ == "__main__":
    run_tests()
