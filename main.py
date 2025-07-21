from embedder import RAGIndexer
from retriever import RAGRetriever
from preprocess import preprocess_for_embedder

def run_indexing():
    indexer = RAGIndexer(model="mistral", index_name="squad")
    preprocess_for_embedder("SQuAD-train-v2.0.json")
    indexer.index_file("preprocessed_dataset.json")

def run_query():
    retriever = RAGRetriever(model="mistral", index_name="squad")
    question = "Beyonce's new single released before the super bowl was called what?" # Answer: "Cadillac Records"
    results, chunk_ids = retriever.search(question, top_k=1)
    context = "\n---\n".join(results)
    answer = retriever.generate_answer(context, question)
    print("\n🧠 Final Answer:\n", answer)
    print("\n🔍 The Retrieved Context:\n", chunk_ids)

def run_query_on_questions():
    retriever = RAGRetriever(model="mistral", index_name="squad")
    question = "Beyonce's new single released before the super bowl was called what?" # Answer: "Solange"
    results, chunk_ids = retriever.search_on_questions(question, top_k=1)
    context = "\n---\n".join(results)
    answer = retriever.generate_answer(context, question)
    print("\n🧠 Final Answer:\n", answer)
    print("\n🔍 The Retrieved Context:\n", chunk_ids)

if __name__ == "__main__":
    print("1. Index documents")
    print("2. Query")
    print("3. Query on questions")

    choice = input("Select an option (1, 2, or 3): ").strip()

    if choice == "1":
        run_indexing()
    elif choice == "2":
        run_query()
    elif choice == "3":
        run_query_on_questions()
    else:
        print("Invalid choice.")
