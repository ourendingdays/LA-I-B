# Custom Modules
from src.rag_visual.lib.semantic_search import SemanticSearch, cosine_similarity
from src.rag_visual.lib.constants import SCORE_PRECISION

# Data Science Libraries
import numpy as np

# Standard Librariestw
import json
import os
import re


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            if not doc.get("description", "").strip():
                continue

            chunks = semantic_chunk(doc["description"], max_chunk_size=4, overlap=1)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs("data/rag_visual/cache", exist_ok=True)
        np.save("data/rag_visual/cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("data/rag_visual/cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("data/rag_visual/cache/chunk_embeddings.npy") and os.path.exists("data/rag_visual/cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("data/rag_visual/cache/chunk_embeddings.npy")
            with open("data/rag_visual/cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
            if self.chunk_embeddings is None:
                raise ValueError("No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

            query_embedding = self.generate_embedding(query)

            # score each chunk
            chunk_scores = []
            for i, chunk_embedding in enumerate(self.chunk_embeddings):
                score = cosine_similarity(query_embedding, chunk_embedding)
                chunk_scores.append({
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": score
                })

            # keep best score per movie
            best_scores = {}
            for cs in chunk_scores:
                movie_idx = cs["movie_idx"]
                if movie_idx not in best_scores or cs["score"] > best_scores[movie_idx]["score"]:
                    best_scores[movie_idx] = cs

            # sort descending, take top limit
            sorted_movies = sorted(best_scores.values(), key=lambda x: x["score"], reverse=True)[:limit]

            # format results
            results = []
            for item in sorted_movies:
                movie = self.documents[item["movie_idx"]]
                results.append({
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                    "score": round(item["score"], SCORE_PRECISION),
                    "metadata": item
                })
            return results


def semantic_chunk(text, max_chunk_size=4, overlap=0):
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    # if only one sentence and no ending punctuation, keep as-is
    if len(sentences) == 1 and not text[-1] in ".!?":
        return [text]

    # strip each sentence, keep only non-empty
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    step = max_chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + max_chunk_size])
        chunks.append(chunk)
        if i + max_chunk_size >= len(sentences):
            break

    return chunks