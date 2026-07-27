# Custom Modules
from src.rag.clients.embedding_client import EmbeddingClient
from src.rag.documents.loaders import load_documents
from src.rag.documents.splitters import split_text_into_chunks

# Data Science Libraries
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Standard Libraries
import os
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

    def list_collections(self) -> List[str]:
        """Returns the names of every collection currently stored."""
        if not self.persist_directory:
            return []  # nothing persisted, so nothing to list
        # Chroma object connects to the same on-disk DB regardless of collection_name;
        # .list_collections() on its underlying client gives us every collection in it.
        temp = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_client.get_langchain_embeddings())
        return [c.name for c in temp._client.list_collections()]

    def get_or_create(self, chunks: List[Document], collection_name: str) -> Chroma:
        """
        Returns a Chroma vector store instance based on the provided collection name and chunks.
        If the collection already exists, it loads it. If not, it creates a new one using the provided chunks.

        Args:
            chunks (List[Document]): List of text chunks to be stored in the vector store.
            collection_name (str): Name of the collection to load or create in the vector store.
        Returns:
            Chroma: Loaded or created vector store instance.
        Behavior:
            collection_name given + already exists  -> loads it (ignores `chunks`)
            collection_name given + doesn't exist    -> creates it from `chunks`
            collection_name not given                -> generates a random name, creates it from `chunks`
        """
        if collection_name and collection_name in self.list_collections():
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_client.get_langchain_embeddings(),
                collection_name=collection_name,
            )
        if collection_name is None:
            collection_name = self.generate_collection_name(prefix="session")

        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_client.get_langchain_embeddings(),
            persist_directory=self.persist_directory,
            collection_name=collection_name,
        )
    
    def generate_collection_name(self, prefix: str = "session") -> str:
        """ Generates a unique collection name using the provided prefix and a random UUID.
        Args:
            prefix (str): Prefix for the collection name. Default is "session". 
        Returns:
            str: Generated unique collection name.
        """
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
