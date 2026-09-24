# Custom Modules
from src.rag_visual.lib.inverted_index import load_movies
from src.rag_visual.lib.hybrid_search import HybridSearch

# Standard Libraries
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open("data/rag_visual/ground_truth_dataset.json", "r") as f:
        golden = json.load(f)
        print(type(golden))
        print(golden.keys() if isinstance(golden, dict) else golden[0])

    documents = load_movies()
    hs = HybridSearch(documents)

    print(f"k={limit}\n")

    for test_case in golden["test_cases"]:
        query = test_case["query"]
        relevant_titles = set(test_case["relevant_docs"])

        results = hs.rrf_search(query, k=60, limit=limit)

        retrieved_titles = [data["doc"]["title"] for _, data in results]

        hits = sum(1 for t in retrieved_titles if t in relevant_titles)
        precision = hits / limit if limit > 0 else 0.0

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_titles)}")
        print()


if __name__ == "__main__":
    main()