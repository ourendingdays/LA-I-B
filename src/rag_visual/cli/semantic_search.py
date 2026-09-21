# Custom Modules
from src.rag_visual.lib.chunked_semantic_search import ChunkedSemanticSearch, semantic_chunk
from src.rag_visual.lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, SemanticSearch

# Standard Libraries
import argparse
import json
import re

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

    chunk_parser = subparsers.add_parser("chunk", help="Chunker simple")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Chunk length")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="How many overlapping words")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunker semantic")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Chunk length")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="How many overlapping words")

    subparsers.add_parser("embed_chunks", help="Build chunk embeddings")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search using chunk embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results")

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
        case "chunk":
            words = args.text.split()
            chunk_size = args.chunk_size
            overlap = args.overlap
            step = chunk_size - overlap
            chunks = []
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i:i + chunk_size]))
                if i + chunk_size >= len(words):
                    break

            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            with open("data/rag_visual/movies.json", "r") as f:
                data = json.load(f)
            documents = data["movies"]
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            with open("data/rag_visual/movies.json", "r") as f:
                data = json.load(f)
            documents = data["movies"]
            css = ChunkedSemanticSearch()
            css.load_or_create_chunk_embeddings(documents)
            results = css.search_chunks(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"    {r['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()