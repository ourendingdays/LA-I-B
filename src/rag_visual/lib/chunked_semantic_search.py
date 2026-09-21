# Custom Modules
from src.rag_visual.lib.semantic_search import SemanticSearch

# Data Science Libraries
import numpy as np

# Standard Libraries
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



def semantic_chunk(text, max_chunk_size=4, overlap=0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if s.strip()]
    step = max_chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + max_chunk_size])
        chunks.append(chunk)
        if i + max_chunk_size >= len(sentences):
            break
    return chunks