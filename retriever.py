import os
import requests
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

class RAGRetriever:
    def __init__(self, model="mistral", index_name="squad"):
        load_dotenv('.env')
        self.model = model
        self.model_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.index_name = index_name
        self.es = Elasticsearch(
            os.getenv("ELASTIC_URL"),
            api_key=os.getenv("ELASTIC_API_KEY"),
            ca_certs=os.getenv("ELASTIC_CA_CERT")
        )

    def get_embedding(self, text):
        response = requests.post(
            f"{self.model_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return response.json()["embedding"]

    def search(self, query_text, top_k=1):
        query_vector = self.get_embedding(query_text)

        script_query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'context_vector') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }

        res = self.es.search(index=self.index_name, body=script_query)
        return [hit["_source"]["context"] for hit in res["hits"]["hits"]], [f"{i}_{hit['_source']['chunk_id']}" for i, hit in enumerate(res["hits"]["hits"]) if i < top_k]

    def generate_answer(self, context, question):
        prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}"
        response = requests.post(
            f"{self.model_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json()["response"]
