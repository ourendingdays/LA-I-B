# Custom Modules
from src.rag_visual.lib.constants import BM25_K1, BM25_B
from src.rag_visual.lib.preprocess import preprocess, tokenize_term
from src.rag_visual.lib.inverted_index import InvertedIndex

# Standard Modules
import argparse   


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # creates the subcommand "search"
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    # adds  positional argument to the subcommand "search", the value we pass after "search" will be captured as "query"
    search_parser.add_argument("query", type=str, nargs="?", default=None, help="Search query")

    # how to add optional arguments
    # parser.add_argument("--test", type=str, nargs="?", const="default", default=None, help="Test the module")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to look up")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term to look up")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to look up")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")

    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(e)
                return
            query_tokens = preprocess(args.query)
            results = []
            for token in query_tokens:
                for doc_id in idx.get_documents(token):
                    movie = idx.docmap[doc_id]
                    if movie not in results:
                        results.append(movie)
                    if len(results) >= 5:
                        break
                if len(results) >= 5:
                    break

            for movie in results:
                print(f"ID: {movie['id']} - {movie['title']}")
        case "build":
            idx = InvertedIndex()
            idx.build()
            idx.save()
            print("Index built and saved.")
        case "tf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(e)
                return
            token = tokenize_term(args.term)
            print(idx.get_tf(args.doc_id, token))
        case "idf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(e)
                return
            token = tokenize_term(args.term)
            idf = idx.get_idf(token)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            idx = InvertedIndex()
            try:
                idx.load()
            except FileNotFoundError as e:
                print(e)
                return
            token = tokenize_term(args.term)
            tf_idf = idx.get_tfidf(args.doc_id, token)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            idx = InvertedIndex()
            idx.load()  # don't forget this
            token = tokenize_term(args.term)
            bm25idf = idx.get_bm25_idf(token)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            idx = InvertedIndex()
            idx.load()
            token = tokenize_term(args.term)
            bm25tf = idx.get_bm25_tf(args.doc_id, token, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case "bm25search":
            idx = InvertedIndex()
            idx.load()
            results = idx.bm25_search(args.query, args.limit)
            for i, (doc_id, score) in enumerate(results, 1):
                title = idx.docmap[doc_id]["title"]
                print(f"{i}. ({doc_id}) {title} — Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()