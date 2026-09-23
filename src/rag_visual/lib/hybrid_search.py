import os

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build()
            self.idx.save()
        else:
            self.idx.load()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        fetch_limit = limit * 500

        # get both result sets
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        # normalize scores
        bm25_scores = self.normalize([score for _, score in bm25_results])
        semantic_scores = self.normalize([r["score"] for r in semantic_results])

        # combine into a dict keyed by doc_id
        combined = {}

        for i, (doc_id, _) in enumerate(bm25_results):
            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": self.idx.docmap[doc_id],
                    "bm25": 0.0,
                    "semantic": 0.0
                }
            combined[doc_id]["bm25"] = bm25_scores[i]

        for i, r in enumerate(semantic_results):
            doc_id = r["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": self.documents[[d["id"] for d in self.documents].index(doc_id)],
                    "bm25": 0.0,
                    "semantic": 0.0
                }
            combined[doc_id]["semantic"] = semantic_scores[i]

        # calculate hybrid scores
        for doc_id in combined:
            combined[doc_id]["hybrid"] = self.hybrid_score(
                combined[doc_id]["bm25"],
                combined[doc_id]["semantic"],
                alpha
            )

        # sort by hybrid score descending
        sorted_results = sorted(combined.items(), key=lambda x: x[1]["hybrid"], reverse=True)

        return sorted_results[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 5) -> list[dict]:
        fetch_limit = limit * 500

        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self.semantic_search.search_chunks(query, fetch_limit)

        combined = {}

        # add BM25 ranks (1-indexed)
        for rank, (doc_id, _) in enumerate(bm25_results, 1):
            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": self.idx.docmap[doc_id],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0
                }
            combined[doc_id]["bm25_rank"] = rank
            combined[doc_id]["rrf_score"] += self.rrf_score(rank, k)

        # add semantic ranks (1-indexed)
        for rank, r in enumerate(semantic_results, 1):
            doc_id = r["id"]
            if doc_id not in combined:
                combined[doc_id] = {
                    "doc": self.documents[[d["id"] for d in self.documents].index(doc_id)],
                    "bm25_rank": None,
                    "semantic_rank": None,
                    "rrf_score": 0.0
                }
            combined[doc_id]["semantic_rank"] = rank
            combined[doc_id]["rrf_score"] += self.rrf_score(rank, k)

        sorted_results = sorted(combined.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        return sorted_results[:limit]

    def rrf_score(self, rank: int, k: int = 60) -> float:
        return 1 / (k + rank)
    
    def hybrid_score(self, bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
        """alpha (or "α") is just a constant that we can use to dynamically control the weighting between the two scores:

        Args:
            bm25_score (float): _description_
            semantic_score (float): _description_
            alpha (float, optional): _description_. Defaults to 0.5.

        Returns:
            float: _description_
        """
        return alpha * bm25_score + (1 - alpha) * semantic_score