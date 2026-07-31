import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from legal_knowledge import get_relevant_legal_knowledge


def run_eval() -> None:
    eval_path = os.path.join(os.path.dirname(__file__), "indian_legal_eval.json")
    with open(eval_path, "r", encoding="utf-8") as handle:
        items = json.load(handle)

    results = []
    for item in items:
        knowledge = get_relevant_legal_knowledge(item["question"])
        matched = 0
        if knowledge:
            for keyword in item["expected_keywords"]:
                if keyword.lower() in knowledge["title"].lower() or keyword.lower() in knowledge["summary"].lower():
                    matched += 1
        score = matched / max(1, len(item["expected_keywords"]))
        results.append({"id": item["id"], "score": score})

    average = sum(result["score"] for result in results) / max(1, len(results))
    print("Evaluation results:")
    for result in results:
        print(f"- {result['id']}: {result['score']:.2f}")
    print(f"Average score: {average:.2f}")


if __name__ == "__main__":
    run_eval()
