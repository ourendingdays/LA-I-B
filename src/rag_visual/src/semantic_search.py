import argparse
import os
import sys
import json

# Own Modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embeddings for the text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embeddings for the query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            sem_search = SemanticSearch()
            with open("data/rag_visual/movies.json", "r") as f:
                data = json.load(f)
            documents = data["movies"]
            sem_search.load_or_create_embeddings(documents)
            results = sem_search.search(args.query, args.limit)
            for i, r in enumerate(results, 1):
                desc = r["description"][:100] + "..." if len(r["description"]) > 100 else r["description"]
                print(f"{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"   {desc}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()