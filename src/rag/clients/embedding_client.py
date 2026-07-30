# Custom Modules
from src.rag.config import load_config

# Data Science Libraries
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Standard Libraries
from typing import List

class EmbeddingClient:
    def __init__(self, model_name: str = None):
        if not model_name:
            config_data = load_config("src/rag/configs/rag_simple.yaml")
            models = config_data.get("sentence_transformer_models", [])
            if not models:
                raise ValueError("No sentence transformer models found in the configuration.")

            model_name = models[0] # or use random, but for now, just use the first one in the list

        self.model_name = model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.model_name)
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