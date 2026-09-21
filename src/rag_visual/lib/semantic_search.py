# Data Science Libraries
import numpy as np
from sentence_transformers import SentenceTransformer

# Standard Libraries
import json
import os


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be None or empty")
        
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        os.makedirs("data/rag_visual/cache", exist_ok=True)
        np.save("data/rag_visual/cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("data/rag_visual/cache/movie_embeddings.npy"):
            self.embeddings = np.load("data/rag_visual/cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, top_k: int = 5):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            scores.append((score, self.documents[i]))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scores[:top_k]:
            results.append({
                "score": score,
                "title": doc["title"],
                "description": doc["description"]
            })
        return results

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    # print(f"Model name: {semantic_search.model.__class__.__name__}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
    return semantic_search

def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    with open("data/rag_visual/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:    {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str):
    print("sdwefbhekwdnfbmeqwnjsaklf,.mnead f;a/edf")
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)