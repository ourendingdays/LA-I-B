from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingClient:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        self._sentence_transformer = None

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """Returns the LangChain-compatible embedding function, for Chroma.from_documents(embedding=...).
        Returns:
            HuggingFaceEmbeddings: The embedding function compatible with LangChain.
        """
        return self.embedding_function

    def encode(self, texts: List[str]) -> np.ndarray:
        """Raw vector encoding for non-LangChain consumers (e.g. FAISS).
        Args:
            texts (List[str]): List of texts to encode.
        Returns:
            np.ndarray: Encoded vectors as a NumPy array.
        """
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer(self.model_name)
        return self._sentence_transformer.encode(texts)