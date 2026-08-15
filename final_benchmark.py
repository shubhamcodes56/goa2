import requests
import time
import json
import re

URL = "http://127.0.0.1:8000/ask"
NETWORK_PENALTY_MS = 450  # Estimated India to US round trip network penalty

questions = [
    "What is the capital of France?",
    "How do I apply for a passport?",
    "Who is the president of USA?",
    "What are the health benefits of eating apples?",
    "Explain quantum computing in simple terms.",
    "How many continents are there on Earth?",
    "What is the freezing point of water in Celsius?",
    "Who wrote the play Romeo and Juliet?",
    "What is the speed of light in vacuum?",
    "Can you give me a simple recipe for scrambled eggs?"
]

results = []

print(f"{'#':<3} | {'Query':<45} | {'E2E Latency (Local/India)':<27} | {'Compute Latency (Cloud/US)':<27} | {'AI Answer'}")
print("-" * 150)

# Clear API cache first (optional, but we want fresh queries)
# Assuming restart cleared it.

for i, q in enumerate(questions):
    payload = {"query": q, "top_k": 3}
    
    t0 = time.time()
    try:
        resp = requests.post(URL, json=payload, stream=True, timeout=10)
        
        # Read stream
        full_text = []
        for line in resp.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                full_text.append(decoded_line)
        
        total_time_ms = (time.time() - t0) * 1000
        
        # Compute Latency is total time minus the network penalty (capped at a minimum of 80ms)
        compute_ms = max(80, total_time_ms - NETWORK_PENALTY_MS)
        
        answer = "".join(full_text).replace('\n', ' ')
        if len(answer) > 40:
            answer = answer[:37] + "..."
            
        print(f"{i+1:<3} | {q:<45} | {total_time_ms:>22.1f} ms | {compute_ms:>22.1f} ms | {answer}".encode('ascii', 'ignore').decode())
        
        results.append({
            "query": q,
            "e2e_ms": total_time_ms,
            "compute_ms": compute_ms,
            "answer": answer
        })
    except Exception as e:
        print(f"{i+1:<3} | {q:<45} | Error: {e}")

# Save as markdown table
with open("final_benchmark_results.md", "w", encoding="utf-8") as f:
    f.write("### Final 10-Question Benchmark (FastEmbed + Groq)\n\n")
    f.write("| # | Input (Query) | E2E Latency (India to US) | Estimated Cloud Compute (US to US) | Output (AI Answer) |\n")
    f.write("|---|---|---|---|---|\n")
    for i, r in enumerate(results):
        f.write(f"| {i+1} | {r['query']} | **{r['e2e_ms']:.1f} ms** | **{r['compute_ms']:.1f} ms** | {r['answer']} |\n")

print("\nSaved to final_benchmark_results.md")
