import json
import os
import requests
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

class RAGIndexer:
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

    def chunk_text(self, text, chunk_size=350, overlap=50):
        words = text.split()
        return [" ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size - overlap)]

    def get_embedding(self, text):
        response = requests.post(
            f"{self.model_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        return response.json()["embedding"]

    def generate_questions(self, context, n_questions=5):
        prompt = f"""You are a helpful assistant that extracts natural, user-like questions from text for use in a Retrieval-Augmented Generation (RAG) system.

                    Your task is to read the following context and generate {n_questions} concise, rephrased user-style questions that could naturally be asked to retrieve information from it.
                    Focus on key facts, events, or named entities in the context.

                    **Important instructions:**

                    - Only return the questions, one per line.

                    - Do not include numbering, explanation, or any additional text.

                    - Each question should be self-contained and reflect how a user might phrase a query.

                    - **Do not add numbers or bullet points before the questions.**

                    **Context:** \n {context}"""
        response = requests.post(
            f"{self.model_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        questions = response.json()["response"].strip().split("\n")
        # print(questions)
        return questions

    def index_file(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_docs = len(data)
        print(f"📄 Total conversations to index: {total_docs}")

        for i, item in enumerate(data, start=1):
            title = item["title"]
            context = item["context"]
            # chunks = self.chunk_text(item["context"]) # The dataset is already chunked

            print(f"➡️  [{i}/{total_docs}] Started Embedding Process 🚩")

            embedding = self.get_embedding(context)
            doc = {
                "title": title,
                "chunk_id": f"{title}_{i}",
                "context": context,
                "context_vector": embedding
            }
            questions = self.generate_questions(context)
            for i, question in enumerate(questions, start=1):
                doc[f"question_{i}_text"] = question
                doc[f"question_{i}_vector"] = self.get_embedding(question)

            self.es.index(index=self.index_name, document=doc)
        print("✅ Indexing complete.")

    