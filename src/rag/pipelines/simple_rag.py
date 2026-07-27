# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.clients.embedding_client import EmbeddingClient
from src.rag.config import load_config
from src.rag.documents.loaders import load_document
from src.rag.documents.splitters import split_text_into_chunks
from src.rag.storage.faiss_index import FaissIndex

# Data Science Libraries
import numpy as np

# Standard Libraries
from pathlib import Path
import random
from typing import List

class SimpleRAG(HuggingFaceClient):
    def __init__(self, embedding_client: EmbeddingClient = None):
        # Initialize the InferenceClient for LLM
        super().__init__()

        embedding_client = embedding_client or EmbeddingClient()
        self.faiss_index = FaissIndex(embedding_client=embedding_client)

    def preprocess_document(self, file_path: Path, query: str, ts_chunk_size: int, ts_chunk_overlap: int, embeddings_top_k: int = 3) -> tuple[List[str], List[float]]:
        """
        Main method to run the RAG pipeline: loads and splits document, creates embeddings and index, performs retrieval of the most relevant chunks.

        Args:
            file_path           (Path) : Path to the document file.
            query               (str)  : User query for retrieval.
            ts_chunk_size       (int)  : Chunk size for splitting the document.
            ts_chunk_overlap    (int)  : Chunk overlap for splitting the document.
            embeddings_top_k    (int)  : Number of top relevant chunks to retrieve. Default is 3.

        Returns:
            tuple: Retrieved chunks (list of str) and distances (list of floats).
        """
        # Loading and spliting the document into chunks
        knowledge_text  = load_document(file_path = file_path)
        chunks, _       = split_text_into_chunks(knowledge_text = knowledge_text,
                                                ts_chunk_size = ts_chunk_size,
                                                ts_chunk_overlap = ts_chunk_overlap)

        # Creating embeddings for the chunks and build a FAISS index
        self.faiss_index.create_embeddings_and_index(chunks = chunks)

        # Performing retrieval based on the user query
        chunks, distances = self.faiss_index.retrieve_chunks(query, top_k = embeddings_top_k)

        return chunks, distances

    def get_embedding_visualization_data(self) -> tuple[List[str], np.ndarray, np.ndarray, List[int]]:
        """
        Returns everything needed to plot the embedding space: all chunk texts, all chunk embeddings, the query embedding,
        and the indices of the retrieved (top-k) chunks. Should be called after preprocess_document().
        """
        if self.faiss_index.get_chunk_embeddings() is None or self.faiss_index.query_embedding is None:
            raise ValueError("No embeddings available. Call preprocess_document() first.")
        return self.faiss_index.get_chunks(), self.faiss_index.get_chunk_embeddings(), self.faiss_index.query_embedding, self.faiss_index.retrieved_indices


if __name__ == "__main__":
    # Loading configuration data
    config_data = load_config("src/rag/configs/rag_simple.yaml")
    PROMPT_TEMPLATE             = config_data['model'].get("llm_prompt")
    MODELS_TO_TEST              = config_data['model'].get("instruct_completion_models", [])
    TEXT_SPLITTER               = config_data.get("text_splitter", {})
    EMBEDDINGS_TOP_K            = config_data.get("embeddings_top_k", 3)
    LLM_MAX_TOKENS              = config_data['model'].get("llm_max_tokens", 200)

    # User can change these values in the Stremlit app, but I hardcode them here to give a sense, as if they come from elsewhere.
    configuration_data = {
        "llm_prompt"                    : PROMPT_TEMPLATE,
        "ts_chunk_size"                 : TEXT_SPLITTER.get("chunk_size", 150),
        "ts_chunk_overlap"              : TEXT_SPLITTER.get("chunk_overlap", 20),
        "embeddings_top_k"              : EMBEDDINGS_TOP_K,
        "llm_max_tokens"                : LLM_MAX_TOKENS
    }

    rag = SimpleRAG()
    #query = "What is the main topic of the document?"  
    query = 'What is the distance from Earth to the Sun?'

    file_path = Path("data/raw/pdf/Full-47.pdf")
    # file_path = Path("data/raw/txt/rag_notebook.txt")
    
    retrieved_chunks, _ = rag.preprocess_document(file_path=file_path, query=query, 
                                                          ts_chunk_size=configuration_data["ts_chunk_size"], 
                                                          ts_chunk_overlap=configuration_data["ts_chunk_overlap"], 
                                                          embeddings_top_k=configuration_data["embeddings_top_k"])
    context = "\n\n".join(retrieved_chunks)

    # Checking for available models and selecting one for the LLM
    available_models = rag.get_working_models(MODELS_TO_TEST)
    model = random.choice(available_models)
    
    answer = rag.ask_model  (query     = query,
                            prompt     = configuration_data["llm_prompt"], 
                            context    = context, 
                            model      = model,
                            max_tokens = configuration_data["llm_max_tokens"])
    print(f"Answer: {answer}")
