"""Batch inference example using Fireworks AI.

Batch API is ideal for:
- Large-scale offline processing
- Cost-sensitive workloads (lower per-token cost)
- Non-latency-sensitive tasks (summarization, classification, embedding)
- Dataset evaluation and benchmarking
"""
from dotenv import load_dotenv
load_dotenv()

from fireworks import Fireworks
import json

client = Fireworks()

# Prepare batch requests (each is a standard chat completion payload)
requests = [
    {
        "custom_id": "req-001",
        "model": "accounts/fireworks/models/qwen3-8b",
        "messages": [
            {"role": "user", "content": "Summarize: The quick brown fox jumps over the lazy dog."}
        ],
        "max_tokens": 100,
    },
    {
        "custom_id": "req-002",
        "model": "accounts/fireworks/models/qwen3-8b",
        "messages": [
            {"role": "user", "content": "Classify the sentiment: I love this product!"}
        ],
        "max_tokens": 50,
    },
    {
        "custom_id": "req-003",
        "model": "accounts/fireworks/models/qwen3-8b",
        "messages": [
            {"role": "user", "content": "Translate to French: Hello, how are you?"}
        ],
        "max_tokens": 100,
    },
]

# Upload batch file
print("Uploading batch requests...")
with open("batch_requests.jsonl", "w") as f:
    for req in requests:
        f.write(json.dumps(req) + "\n")

# Create batch job (uncomment to run with a valid API key)
# batch = client.batches.create(
#     input_file="batch_requests.jsonl",
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
# )
# print(f"Batch created: {batch.id}")
# print(f"Status: {batch.status}")

# Check batch status
# batch_status = client.batches.retrieve(batch.id)
# print(f"Progress: {batch_status.request_counts.completed}/{batch_status.request_counts.total} completed")

# Retrieve results when complete
# results = client.files.content(batch_status.output_file_id)
# for line in results:
#     data = json.loads(line)
#     print(f"{data['custom_id']}: {data['response']['body']['choices'][0]['message']['content']}")

print("\nNote: Uncomment the batch creation code to run with a valid API key.")
print("Batch API docs: https://docs.fireworks.ai/guides/batch-inference")
