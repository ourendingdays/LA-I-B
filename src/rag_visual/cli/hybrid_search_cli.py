# Custom Modules
from src.rag_visual.lib.inverted_index import load_movies
from src.rag_visual.lib.hybrid_search import HybridSearch
from src.rag_visual.lib.query_enhance import spell_correct, rewrite_query, expand_query, rerank_individual, rerank_batch, rerank_cross_encoder

# Standard Libraries
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalise_parser = subparsers.add_parser("normalize", help="Score normalization")
    normalise_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    # hybrid weighted search
    weighted_search = subparsers.add_parser("weighted-search", help="Weighted Search")
    weighted_search.add_argument("query", type=str, help="Text query for the weighted search")
    weighted_search.add_argument("--alpha", type=float, default = 0.5,help="Alpha constant for weighting between BM25 and semantic scores")
    weighted_search.add_argument("--limit", type=int, default=5, help="Limit the number of search results returned")

    # hybrid reciprocal rank fusion search
    rrf_parser = subparsers.add_parser("rrf-search", help="RRF hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    
    # hybrid reciprocal rank fusion search with LLM
    rrf_parser_llm = subparsers.add_parser("rrf-search-llm", help="RRF hybrid search with LLM")
    rrf_parser_llm.add_argument("query", type=str, help="Search query")
    rrf_parser_llm.add_argument("-k", type=int, default=60, help="RRF k parameter")
    rrf_parser_llm.add_argument("--limit", type=int, default=5, help="Number of results")
    rrf_parser_llm.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser_llm.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method",
    )

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
        case "rrf-search":
            documents = load_movies()
            hs = HybridSearch(documents)
            results = hs.rrf_search(args.query, args.k, args.limit)

            for i, (doc_id, data) in enumerate(results, 1):
                bm25_rank = data["bm25_rank"] or "N/A"
                semantic_rank = data["semantic_rank"] or "N/A"
                print(f"{i}. {data['doc']['title']}")
                print(f"   RRF Score: {data['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {data['doc']['description'][:100]}")
        case "rrf-search-llm":
            documents = load_movies()
            hs = HybridSearch(documents)

            query = args.query
            if args.enhance == "spell":
                enhanced = spell_correct(query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            elif args.enhance == "rewrite":
                enhanced = rewrite_query(query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            elif args.enhance == "expand":
                enhanced = expand_query(query)
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = f"{query} {enhanced}"

            if args.rerank_method == "individual":
                results = hs.rrf_search(query, args.k, args.limit * 5)
                results = rerank_individual(query, results, args.limit)
            elif args.rerank_method == "batch":
                results = hs.rrf_search(query, args.k, args.limit * 5)
                results = rerank_batch(query, results, args.limit)
            elif args.rerank_method == "cross_encoder":
                results = hs.rrf_search(query, args.k, args.limit * 5)
                results = rerank_cross_encoder(query, results, args.limit)
            else:
                results = hs.rrf_search(query, args.k, args.limit)

            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            for i, (doc_id, data) in enumerate(results, 1):
                bm25_rank = data["bm25_rank"] or "N/A"
                semantic_rank = data["semantic_rank"] or "N/A"
                print(f"{i}. {data['doc']['title']}")
                if "rerank_score" in data:
                    print(f"   Re-rank Score: {data['rerank_score']:.3f}/10")
                if "rerank_rank" in data:
                    print(f"   Re-rank Rank: {data['rerank_rank']}")
                if "cross_encoder_score" in data:
                    print(f"   Cross Encoder Score: {data['cross_encoder_score']:.3f}")
                print(f"   RRF Score: {data['rrf_score']:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {data['doc']['description'][:100]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()