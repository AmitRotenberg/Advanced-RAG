from embedder import RAGIndexer
from retriever import RAGRetriever
from preprocess import preprocess_for_embedder

def run_indexing():
    indexer = RAGIndexer(model="mistral", index_name="squad")
    preprocess_for_embedder("SQuAD-train-v2.0.json")
    indexer.index_file("preprocessed_dataset.json")

def run_query():
    retriever = RAGRetriever(model="mistral", index_name="squad")
    question = "After what movie portraying Etta James, did Beyonce create Sasha Fierce?" # Answer: "Cadillac Records"
    results, chunk_ids = retriever.search(question, top_k=1)
    context = "\n---\n".join(results)
    answer = retriever.generate_answer(context, question)
    print("\n🧠 Final Answer:\n", answer)
    print("\n🔍 The Retrieved Context:\n", chunk_ids)
if __name__ == "__main__":
    print("1. Index documents")
    print("2. Query and answer")
    choice = input("Select an option (1 or 2): ").strip()

    if choice == "1":
        run_indexing()
    elif choice == "2":
        run_query()
    else:
        print("Invalid choice.")
