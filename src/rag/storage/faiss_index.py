# Custom Libraries
from src.rag.clients.embedding_client import EmbeddingClient

# Data Science Libraries
import faiss
import numpy as np

# Standard Libraries
from typing import List, Tuple



class FaissIndex:
    def __init__(self, embedding_client: EmbeddingClient = None):
        self.embedding_client = embedding_client or EmbeddingClient()
        self.index              = None
        self.chunks             = None
        self.chunk_embeddings   = None
        self.query_embedding    = None   
        self.retrieved_indices  = None

    def create_embeddings_and_index(self, chunks : List[str]):
        """
        Creates embeddings for the given chunks and builds a FAISS index.

        Args:
            chunks      (list)  : List of text chunks.
        """
        # Embedding all chunks. This will take a moment as the model "reads" and "understands" each chunk.
        chunk_embeddings = self.embedding_client.encode(chunks) 

        # dimension of our vectors are 384 (the size of the embedding model output)
        dims = chunk_embeddings.shape[1]

        # Creating the FAISS-db index with IndexFlatL2 to store the embeddings in a vector-index - the simplest, most basic index, 
        # that calculates the exact distance (L2 distance) between our query and all vectors.
        self.index = faiss.IndexFlatL2(dims)

        # Adding our chunk embeddings to the index. We must convert to float32 for FAISS
        self.index.add(np.array(chunk_embeddings).astype('float32'))

        self.chunks = chunks
        self.chunk_embeddings = np.array(chunk_embeddings).astype('float32')

    def retrieve_chunks(self, query: str, top_k: int = 3) -> tuple[List[str], List[float]]:
        """
        Retrieves the top_k most relevant chunks based on the user query.

        Args:
            query (str): User query for retrieval.
            top_k (int): Number of top relevant chunks to retrieve. Default is 3.

        Returns:
            tuple: Retrieved chunks (list of str) and distances (list of floats).
        """
        # Create embedding for the user query
        self.query_embedding = self.embedding_client.encode([query]).astype('float32')

        # Search in the FAISS index for the top_k most similar chunks
        distances, indices = self.index.search(self.query_embedding, top_k)
        self.retrieved_indices = indices[0].tolist()

        # Retrieve the actual text chunks based on the indices
        retrieved_chunks = [self.chunks[i] for i in indices[0]]

        # Print the retrieved chunks and their distances
        # for i, chunk in enumerate(retrieved_chunks):
        #     print(f"Chunk {i+1} (Distance: {distances[0][i]}):\n{chunk}\n")
        return retrieved_chunks, distances[0]
    

    def get_index(self) -> faiss.IndexFlatL2:
        """
        Returns the FAISS index.

        Returns:
            faiss.IndexFlatL2: The FAISS index.
        """
        return self.index

    def get_chunk_embeddings(self) -> np.ndarray:
        """
        Returns the chunk embeddings.

        Returns:
            np.ndarray: The chunk embeddings.
        """
        return self.chunk_embeddings

    def get_chunks(self) -> List[str]:
        """
        Returns the list of chunks.

        Returns:
            list: The list of chunks.
        """
        return self.chunks