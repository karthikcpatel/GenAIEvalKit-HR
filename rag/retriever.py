import csv
from evaluator.similarity import similarity_score

# Load HR knowledge base from CSV
KB = []


def load_kb(csv_path="prompts/hr_prompt_suite.csv"):
    global KB

    if KB:
        return  # already loaded

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                KB.append({
                    "question": row["prompt"],
                    "answer": row["expected_answer"],
                    "category": row["category"],
                    "key_points": row.get("key_points", "")
                })
    except Exception as e:
        print(f"[Retriever Error] Failed to load KB: {e}")


def retrieve(query, top_k=3):
    """
    Retrieve top-k relevant entries using semantic similarity.
    """
    load_kb()

    scored = []

    for item in KB:
        score = similarity_score(query, item["question"])
        scored.append((item, score))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[1], reverse=True)

    top_results = scored[:top_k]

    return {
        "contexts": [r[0]["answer"] for r in top_results],
        "questions": [r[0]["question"] for r in top_results],
        "categories": [r[0]["category"] for r in top_results],
        "scores": [r[1] for r in top_results]
    }