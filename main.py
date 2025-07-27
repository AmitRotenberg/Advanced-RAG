from embedder import RAGIndexer
from retriever import RAGRetriever
from preprocess import preprocess_for_embedder

def run_indexing():
    indexer = RAGIndexer(model="mistral", index_name="squad")
    preprocess_for_embedder("SQuAD-train-v2.0.json")
    indexer.index_file("preprocessed_dataset.json")

def run_query(query, type):
    retriever = RAGRetriever(model="mistral", index_name="squad")
    if type == "1":
        results, chunk_ids = retriever.search(query, top_k=1)
    elif type == "2":
        results, chunk_ids = retriever.search_on_questions(query, top_k=1)
    context = "\n---\n".join(results)
    answer = retriever.generate_answer(context, query)
    print("\n🧠 Final Answer:\n", answer)
    print("\n🔍 The Retrieved Context:\n", chunk_ids)

# def run_query_on_questions(query):
#     retriever = RAGRetriever(model="mistral", index_name="squad")
#     results, chunk_ids = retriever.search_on_questions(query, top_k=1)
#     context = "\n---\n".join(results)
#     answer = retriever.generate_answer(context, query)
#     print("\n🧠 Final Answer:\n", answer)
#     print("\n🔍 The Retrieved Context:\n", chunk_ids)

if __name__ == "__main__":
    print("1. Index documents")
    print("2. Query")

    choice = input("Select an option (1, 2, or 3): ").strip()

    if choice == "1":
        run_indexing()
    elif choice == "2":
        print("1. Regular")
        print("2. Advanced")
        type = input("Select an option (1, 2): ").strip()
        query = input("Enter your query: ")
        run_query(query, type)
    
    else:
        print("Invalid choice.")
