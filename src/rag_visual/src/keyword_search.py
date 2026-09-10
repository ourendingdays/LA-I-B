import argparse
import json
import string
from nltk.stem import PorterStemmer
import os
cwd = os.getcwd()
print(cwd)


def remove_punctuation(text:str) -> str:
    # string.punctuation is just a string containing all punctuation characters: !"#$%&'()*+,-./:;<=>?@[\]^_{|}~
    # str.maketrans("", "", string.punctuation) builds a translation table that maps every punctuation character to None (i.e. delete it)
    # .translate(table) applies that table to the string
    return text.translate(str.maketrans("", "", string.punctuation))


def load_stop_words(file_path: str) -> set:
    with open(file_path) as f:
        stop_words = f.read().splitlines()
    # preprocess each stop word the same way you preprocess query/title tokens
    return set(remove_punctuation(word.lower()) for word in stop_words)

# load once at module level so you don't re-read the file every call
STOP_WORDS = load_stop_words("data/rag_visual/stop_words.txt")


def preprocess(query: str) -> str:
    # Remove Case Sensitivity
    lower_ = query.lower()
    # Remove Punctuation
    punc_ = remove_punctuation(lower_)
    # Tokenization
    token_ = punc_.split()
    # Stop Words
    token_ = [word for word in token_ if word not in STOP_WORDS]
    # stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in token_]
    return stemmed

def search_movies(query: str) -> None:
    with open("data/rag_visual/movies.json", "r") as file:
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

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    # creates the subcommand "test" itself
    test_parser = subparsers.add_parser("test", help="Test the module")
    # adds a positional argument to the subcommand "test", the valeu we can pass after test
    test_parser.add_argument("test", type=str, nargs="?", default=None, help="Optional test value")

    # optional arguments
    # parser.add_argument("--test", type=str, nargs="?", const="default", default=None, help="Test the module")

    args = parser.parse_args()

    match args.command:
        case "test":
            print(f"Test value: {args.test}")
            res = search_movies(query = "Great")
            print(res)
        case "search":
            print(f"Searching for: {args.query}")
            res = search_movies(query=args.query)
            for i, movie_title in enumerate(res):
                print(f"{i}. {movie_title['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()