from sentence_transformers import SentenceTransformer
import numpy as np
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
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
