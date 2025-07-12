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

# --- Get Embedding from Ollama ---
def get_embedding(text, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]

# --- Vector Search ---
def search_similar(query_text, top_k=5):
    query_vector = get_embedding(query_text)

    script_query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'conversation_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }

    res = es.search(index="calls", body=script_query)
    return [hit["_source"]["conversation"] for hit in res["hits"]["hits"]]

# --- LLM Generation ---
def generate_answer(context, question, model="mistral"):
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# --- Run ---
if __name__ == "__main__":
    question = "Give me summary of water related issues"
    results = search_similar(question, top_k=5)
    context = "\n---\n".join(results)
    answer = generate_answer(context, question)
    print("\n🧠 Final Answer:\n", answer)
