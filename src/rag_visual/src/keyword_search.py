import argparse
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.preprocess import preprocess
from lib.inverted_index import InvertedIndex

def search_movies(query: str) -> None:
    with open("data/movies.json", "r") as file:
        data = json.load(file)

    result = []
    query = preprocess(query)
    for movie in data['movies']:
        if any(
            any(query_token in title_token for title_token in preprocess(movie['title']))
            for query_token in query
        ):
            result.append(movie)
    return result
   
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # creates the subcommand "search"
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    # adds  positional argument to the subcommand "search", the value we pass after "search" will be captured as "query"
    search_parser.add_argument("query", type=str, nargs="?", default=None, help="Search query")

    # optional arguments
    # parser.add_argument("--test", type=str, nargs="?", const="default", default=None, help="Test the module")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()