# Custom Modules
from src.rag.clients.embedding_client import EmbeddingClient
from src.rag.documents.loaders import load_documents
from src.rag.documents.splitters import split_text_into_chunks

# Data Science Libraries
from langchain_chroma import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Standard Libraries
import os
from pathlib import Path
from typing import List, Optional
import uuid


class ChromaStore:
    def __init__(self, embedding_client: EmbeddingClient, persist_directory: Optional[str] = None):
        """
        Initializes the ChromaStore with an embedding client and an optional persistence directory.
        Args:
            embedding_client (EmbeddingClient): An instance of the EmbeddingClient to generate embeddings.
            persist_directory (Optional[str]): Directory path for persisting the vector store.
        If persist_directory is None, the vector store will be ephemeral (in-memory) and will not be saved to disk: AgenticRAG-style, one-off session. 
        If a path is provided, the vector store will be persisted to that location, allowing it to be reused across different runs of the application : VectorSearchRAG-style, reused across runs.
        """
        self.embedding_client = embedding_client
        self.persist_directory = persist_directory

    def create_vector_store(self, chunks: List[Document], collection_name: str = None) -> Chroma:
        """
        Creates a vector store from the provided text chunks using Chroma and HuggingFaceEmbeddings.

        Args:
            chunks (List[Document]): List of text chunks to be stored in the vector store.
            collection_name (str): Name of the collection to create in the vector store.
        Returns:
            Chroma: Created vector store instance.
        """
        if collection_name is None:
            collection_name = self.generate_collection_name(prefix="session")

        return Chroma.from_documents(
            documents           = chunks,
            embedding           = self.embedding_client.get_langchain_embeddings(),
            persist_directory   = self.persist_directory, # saves the database in a folder
            collection_name     = collection_name
        )

    def load_vector_store(self, collection_name: str) -> Chroma:
        """
        Loads an existing vector store from the specified directory using Chroma and HuggingFaceEmbeddings.
        
        Args:
            collection_name (str): Name of the collection to load from the vector store.
        Returns:
            Chroma: Loaded vector store instance.
        """
        if self.persist_directory is None:
            raise ValueError("Cannot load a collection without a persist_directory.")
        
        return Chroma(
            persist_directory   = self.persist_directory,
            embedding_function  = self.embedding_client.get_langchain_embeddings(),
            collection_name     = collection_name
        )

    def get_vector_store(self):
        """
        Returns the vector store instance. If it hasn't been initialized yet, raises an error.

        Returns:
            Chroma: The vector store instance.
        """
        if self.vector_store is None:
            raise ValueError("Vector store has not been initialized. Please call 'initialize_vector_store' first.")
        return self.vector_store

    def get_or_create(self, chunks: List[Document], collection_name: str) -> Chroma:
        """VectorSearchRAG's 'initialize_vector_store' pattern, generalized.
        Args:
            chunks (List[Document]): List of text chunks to be stored in the vector store.
            collection_name (str): Name of the collection to create or load in the vector store.
        Returns:
            Chroma: Created or loaded vector store instance.
        """
        if self.persist_directory and self._collection_exists():
            return self.load_vector_store(collection_name=collection_name)
        return self.create_vector_store(chunks, collection_name=collection_name)
    
    def generate_collection_name(self, prefix: str = "session") -> str:
        """ Generates a unique collection name using the provided prefix and a random UUID.
        Args:
            prefix (str): Prefix for the collection name. Default is "session". 
        Returns:
            str: Generated unique collection name.
        """
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _collection_exists(self) -> bool:
        # Simplification: checks if the persist directory has been initialized at all.
        # A more precise check would query Chroma's client for existing collection names.
        return os.path.exists(self.persist_directory)
