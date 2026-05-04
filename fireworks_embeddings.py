"""Text embedding example using Fireworks AI.

Query and document embeddings are useful for:
- Semantic search
- Document clustering
- Similarity matching
- RAG retrieval pipelines
"""
from dotenv import load_dotenv
load_dotenv()

from fireworks import Fireworks

client = Fireworks()

# Single embedding
response = client.embeddings.create(
    model="accounts/fireworks/models/nomic-embed-text-v1",
    input="Fireworks AI provides fast inference for open-source models",
)

print("Embedding dimension:", len(response.data[0].embedding))
print("First 5 values:", response.data[0].embedding[:5])

# Batch embedding (more efficient)
query = "What is the weather in Paris?"
documents = [
    "Paris has mild winters and warm summers with occasional rain.",
    "The Eiffel Tower is a famous landmark in Paris, France.",
    "Paris is the capital and most populous city of France.",
]

all_texts = [query] + documents
batch_response = client.embeddings.create(
    model="accounts/fireworks/models/nomic-embed-text-v1",
    input=all_texts,
)

# Compute cosine similarities
import math

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)

query_embedding = batch_response.data[0].embedding
doc_embeddings = [e.embedding for e in batch_response.data[1:]]

print("\nQuery:", query)
for doc, emb in zip(documents, doc_embeddings):
    sim = cosine_similarity(query_embedding, emb)
    print(f"  Similarity {sim:.4f} — {doc[:50]}...")
