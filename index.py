import json
import os
import requests
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv('.env')

# Elasticsearch client
es = Elasticsearch(
    os.getenv("ELASTIC_URL"),
    api_key=os.getenv("ELASTIC_API_KEY"),
    ca_certs=os.getenv("ELASTIC_CA_CERT")
)

# --- Chunking ---
def chunk_text(text, chunk_size=350, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# --- Get Embedding from Ollama ---
def get_embedding(text, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]

# --- Index Documents ---
def index_documents(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        conversation_id = item["conversation_id"]
        chunks = chunk_text(item["conversation"])

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            doc = {
                "conversation_id": conversation_id,
                "chunk_id": i,
                "conversation": chunk,
                "conversation_vector": embedding
            }
            es.index(index="calls", document=doc)

    print("✅ Finished indexing.")

# --- Run ---
if __name__ == "__main__":
    index_documents("conversations.json")
