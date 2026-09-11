import json
import math
import os
import pickle
from typing import Counter

# Own Modules
from lib.preprocess import preprocess

class InvertedIndex:
    def __init__(self):
        self.index = {}             # token -> set of doc IDs
        self.docmap = {}            # doc ID -> full movie object
        self.term_frequencies = {}  # doc_id -> Counter

    def __add_document(self, doc_id, text):
        """Tokenizes the text with your existing preprocess, then maps each token to a set of doc IDs.
        
        Args:
            doc_id (str): The unique identifier for the document.
            text (str): The text content of the document to be indexed.
        """
        tokens = preprocess(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        """Preprocesses the search term and looks it up in the index.

        Args:
            term (str): The search term to look up in the index.

        Returns:
            list: A sorted list of document IDs that match the search term.
        """
        processed = preprocess(term)
        if not processed:
            return []
        token = processed[0]
        return sorted(self.index.get(token, []))

    def get_tf(self, doc_id, term):
        """Returns the term frequency of a term in a specific document.

        Args:
            doc_id (str): The unique identifier for the document.
            term (str): The term whose frequency is to be retrieved.

        Returns:
            int: The frequency of the term in the document. Returns 0 if the document or term is not found.
        """
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id].get(term, 0)

    def get_idf(self, term):
        total_docs = len(self.docmap)
        docs_with_term = len(self.index.get(term, set()))
        if docs_with_term == 0:
            return 0
        return math.log(total_docs / (1 + docs_with_term))

    def get_tfidf(self, doc_id, term):
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def build(self):
        """Concatenates title and description as the exercise specifies."""
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)

    def save(self):
        """Pickles both dictionaries to cache/."""
        os.makedirs("data/rag_visual/cache", exist_ok=True)
        with open("data/rag_visual/cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("data/rag_visual/cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("data/rag_visual/cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        """Loads the index and docmap from cache files.

        Raises:
            FileNotFoundError: If the cache files do not exist.
        """
        if not os.path.exists("data/rag_visual/cache/index.pkl") or not os.path.exists("data/rag_visual/cache/docmap.pkl"):
            raise FileNotFoundError("Index files not found. Run 'build' first.")
        with open("data/rag_visual/cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("data/rag_visual/cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("data/rag_visual/cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)


def load_movies():
    with open("data/rag_visual/movies.json", "r") as file:
        data = json.load(file)
    return data["movies"]