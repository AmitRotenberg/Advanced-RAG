import json

def preprocess_for_embedder(filepath, n=66, output_path="preprocessed_dataset.json"):
    results = []
    count = 0

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data["data"]:
        title = item.get("title", "")
        for paragraph in item.get("paragraphs", []):
            context = paragraph.get("context", "")
            results.append({"title": title, "context": context})
            count += 1
            if count >= n:
                break
        if count >= n:
            break

    # Write results to a new JSON file
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)

    return results

# preprocess_for_embedder("SQuAD-train-v2.0.json")