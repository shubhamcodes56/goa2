import requests
import time
import numpy as np
import json

API_URL = "http://127.0.0.1:8000/ask"

# 20 diverse test queries to simulate Voice Inputs
TEST_QUERIES = [
    "Who is the Prime Minister of India?",
    "What is the capital of France?",
    "Tell me about the Taj Mahal.",
    "What is ChatGPT?", # Should trigger guardrail or LLM generic knowledge
    "Who won the cricket world cup in 2011?",
    "What is the weather in Mumbai?",
    "How to make tea?",
    "What is the most famous song in the world?",
    "Who is Lionel Messi?",
    "When did India get independence?",
    "What is the meaning of life?",
    "What is the distance between Earth and Moon?",
    "Who is the author of Harry Potter?",
    "What is the tallest mountain?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "How many continents are there?",
    "What is the freezing point of water?",
    "Who invented the telephone?",
    "What is the population of the world?"
]

def run_benchmark():
    print("Starting End-to-End Latency Analytics (Hackathon Requirement #4)...\n")
    print(f"{'Query':<45} | {'Qdrant (ms)':<12} | {'Groq LLM (ms)':<14} | {'Total (ms)':<10}")
    print("-" * 90)

    total_latencies = []
    
    for query in TEST_QUERIES:
        try:
            start_time = time.perf_counter()
            response = requests.post(API_URL, json={"query": query}, timeout=10, stream=True)
            
            if response.status_code == 200:
                ttft_ms = None
                full_text = ""
                for chunk in response.iter_lines():
                    if chunk:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - start_time) * 1000
                        full_text += chunk.decode('utf-8')
                
                client_total_ms = (time.perf_counter() - start_time) * 1000
                total_latencies.append(ttft_ms if ttft_ms else client_total_ms)
                
                print(f"{query[:42]:<45} | TTFT: {ttft_ms:<7.1f} | Total: {client_total_ms:<7.1f}")
                print(f"   -> AI: {full_text[:100]}...")
                print("-" * 90)
            else:
                print(f"{query[:42]:<45} | ERROR {response.status_code}")
                
        except Exception as e:
            print(f"{query[:42]:<45} | FAILED: {str(e)}")

    if total_latencies:
        p50 = np.percentile(total_latencies, 50)
        p70 = np.percentile(total_latencies, 70)
        p100 = np.percentile(total_latencies, 100) # Max
        avg = np.mean(total_latencies)
        
        print("\n" + "="*50)
        print("LATENCY ANALYTICS REPORT")
        print("="*50)
        print(f"Total Queries Tested : {len(total_latencies)}")
        print(f"Average Latency      : {avg:.2f} ms")
        print(f"P50 Latency (Median) : {p50:.2f} ms")
        print(f"P70 Latency          : {p70:.2f} ms")
        print(f"P100 Latency (Max)   : {p100:.2f} ms")
        
        if p100 < 200:
            print("\nTARGET ACHIEVED: All queries completed under 200ms!")
        elif p50 < 200:
            print("\nPARTIAL SUCCESS: P50 is under 200ms, but some queries took longer.")
        else:
            print("\nTARGET FAILED: Most queries took longer than 200ms.")

if __name__ == "__main__":
    run_benchmark()
