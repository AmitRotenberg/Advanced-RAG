import json
import os
from tqdm import tqdm
from dotenv import load_dotenv

# import your retriever class
from retriever import RAGRetriever

def evaluate_method(retriever, data, method_name="search", top_k=1, max_chunks=None):
    print(f"=== Evaluating method '{method_name}' (top_k={top_k}, max_chunks={max_chunks}) ===")
    total = correct = 0
    chunk_cnt = 0

    for entry in data["data"]:
        title = entry.get("title", "<no-title>")
        for para in entry["paragraphs"]:
            # stop once we’ve seen max_chunks
            if max_chunks and chunk_cnt >= max_chunks:
                print(f"Reached max_chunks ({max_chunks}), stopping evaluation.")
                return correct / total if total else 0, total

            chunk_cnt += 1
            # Determine the ground truth chunk_id based on indexing
            true_chunk_id = f"{title}_{chunk_cnt}"
            print(f"\n-- Chunk {chunk_cnt}/{max_chunks if max_chunks else '?'} "
                  f"(title: '{title}', id: '{true_chunk_id}') --")

            for qa in para["qas"]:
                qid = qa.get("id", "<no-id>")
                if qa.get("is_impossible", False):
                    print(f"   [Skipping impossible question id={qid}]")
                    continue

                total += 1
                question = qa["question"]
                print(f"   [Question #{total} id={qid}] '{question}'")

                # perform retrieval
                retrieve_fn = getattr(retriever, method_name)
                ctxs, retrieved_ids = retrieve_fn(question, top_k=top_k)

                # log retrieved chunk ids
                print(f"      Retrieved chunk IDs: {retrieved_ids}")

                # check if the true chunk id was returned
                match = true_chunk_id in retrieved_ids
                print(f"      -> Chunk match: {match}")
                if match:
                    correct += 1

    accuracy = correct / total if total else 0
    print(f"Finished evaluating '{method_name}': accuracy={accuracy:.2%} ({correct}/{total})")
    return accuracy, total


def main():
    load_dotenv()  # loads .env to point to your ES + Ollama
    squad_path = "SQuAD-train-v2.0.json"

    # load the SQuAD file
    with open(squad_path, "r", encoding="utf-8") as f:
        squad = json.load(f)

    # instantiate your retriever
    retriever = RAGRetriever(
        model=os.getenv("OLLAMA_MODEL", "mistral"),
        index_name=os.getenv("ES_INDEX", "squad")
    )

    # evaluate both methods (with logging inside evaluate_method)
    acc_ctx, total = evaluate_method(
        retriever, squad,
        method_name="search", top_k=10,
        max_chunks=10
    )
    acc_qs, _ = evaluate_method(
        retriever, squad,
        method_name="search_on_questions", top_k=10,
        max_chunks=10
    )

    print("\n=== Summary ===")
    print(f"Context-only retrieval accuracy:   {acc_ctx:.2%} ({int(acc_ctx*total)}/{total})")
    print(f"Question-based retrieval accuracy: {acc_qs:.2%} ({int(acc_qs*total)}/{total})")

if __name__ == "__main__":
    main()
