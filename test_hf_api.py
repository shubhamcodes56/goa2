import requests
import time

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def query(payload):
	response = requests.post(API_URL, json=payload)
	return response.json()

t0 = time.time()
output = query({"inputs": "What is the capital of France?"})
print(f"Time: {time.time() - t0}")
print(len(output))
