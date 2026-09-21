import json
import math
import os
import pickle
from typing import Counter

# Own Modules
from lib.constants import BM25_K1, BM25_B, CACHE_DIR
from lib.preprocess import preprocess, tokenize_term

class InvertedIndex:
    def __init__(self):
        self.index = {}             # token -> set of doc IDs
        self.docmap = {}            # doc ID -> full movie object
        self.term_frequencies = {}  # doc_id -> Counter
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = preprocess(text)
        self.doc_lengths[doc_id] = len(tokens)
        
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

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

    def get_bm25_idf(self, term: str) -> float:
        n = len(self.docmap)
        df = len(self.index.get(term, set()))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def bm25_tf_command(doc_id, term, k1=BM25_K1):
        idx = InvertedIndex()
        idx.load()
        token = tokenize_term(term)
        return idx.get_bm25_tf(doc_id, token, k1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_doc_len = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_len / avg_doc_len) if avg_doc_len > 0 else 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit=5):
        query_tokens = preprocess(query)
        scores = {}

        for doc_id in self.docmap:
            score = 0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            if score > 0:
                scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:limit]

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

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
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

def load_movies():
    with open("data/rag_visual/movies.json", "r") as file:
        data = json.load(file)
    return data["movies"]