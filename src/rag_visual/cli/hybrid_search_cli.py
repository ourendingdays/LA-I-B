# Custom Modules
from src.rag_visual.lib.inverted_index import load_movies
from src.rag_visual.lib.hybrid_search import HybridSearch

# Standard Libraries
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalise_parser = subparsers.add_parser("normalize", help="Score normalization")
    normalise_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    weighted_search = subparsers.add_parser("weighted-search", help="Weighted Search")
    weighted_search.add_argument("query", type=str, help="Text query for the weighted search")
    weighted_search.add_argument("--alpha", type=float, default = 0.5,help="Alpha constant for weighting between BM25 and semantic scores")
    weighted_search.add_argument("--limit", type=int, default=5, help="Limit the number of search results returned")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return
            min_score = min(scores)
            max_score = max(scores)
            if min_score == max_score:
                for _ in scores:
                    print(f"* 1.0000")
            else:
                for score in scores:
                    normalized = (score - min_score) / (max_score - min_score)
                    print(f"* {normalized:.4f}")
        case "weighted-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.weighted_search(args.query, args.alpha, args.limit)

            for i, (doc_id, data) in enumerate(results, 1):
                print(f"{i}. {data['doc']['title']}")
                print(f"   Hybrid Score: {data['hybrid']:.3f}")
                print(f"   BM25: {data['bm25']:.3f}, Semantic: {data['semantic']:.3f}")
                print(f"   {data['doc']['description'][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()