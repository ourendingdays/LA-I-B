# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.config import load_config
from src.rag.documents.loaders import load_document
from src.rag.documents.splitters import split_text_into_chunks

# Data Science Libraries
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Standard Libraries
from pathlib import Path
import random
from typing import Dict, List

class SimpleRAG(HuggingFaceClient):
    def __init__(self):
        # Initialize the InferenceClient for LLM
        super().__init__()

        self.chunk_embeddings = None
        self.query_embedding = None
        self.retrieved_indices = None

    def preprocess_document(self, file_path: Path, query: str, configuration_data: Dict) -> tuple[List[str], List[float]]:
        """
        Main method to run the RAG pipeline: loads and splits document, creates embeddings and index, performs retrieval of the most relevant chunks.

        Args:
            file_path           (Path) : Path to the document file.
            query               (str)  : User query for retrieval.
            configuration_data  (dict) : Configuration data loaded from YAML file.

        Returns:
            tuple: Retrieved chunks (list of str) and distances (list of floats).
        """
        # Load and split the document into chunks
        knowledge_text  = load_document(file_path = file_path)
        chunks, _       = split_text_into_chunks(knowledge_text = knowledge_text,
                                        ts_chunk_size = configuration_data["ts_chunk_size"],
                                        ts_chunk_overlap = configuration_data["ts_chunk_overlap"])

        # Create embeddings for the chunks and build a FAISS index
        self.index, self.chunks = self.create_embeddings_and_index(chunks = chunks, model_name = configuration_data.get("sentence_transformer_model", "all-MiniLM-L6-v2"))

        # Perform retrieval based on the user query
        chunks, distances = self.retrieve_chunks(query, top_k = configuration_data.get("embeddings_top_k", 3))

        return chunks, distances

    def create_embeddings_and_index(self, chunks : List[str], model_name : str ="all-MiniLM-L6-v2") -> tuple[faiss.IndexFlatL2, List[str]] :
        """
        Creates embeddings for the given chunks and builds a FAISS index.

        Args:
            chunks      (list)  : List of text chunks.
            model_name  (str)   : Name of the embedding model to use. Default is "all-MiniLM-L6-v2" - small embedding model, that runs 100% even on not the most demanding local machines.
        Returns:
            tuple: FAISS index and the list of chunks.
        """
        
        self.embedding_model = SentenceTransformer(model_name)

        # Embedding all chunks. This will take a moment as the model "reads" and "understands" each chunk.
        chunk_embeddings = self.embedding_model.encode(chunks)

        # dimension of our vectors are 384 (the size of the embedding model output)
        dims = chunk_embeddings.shape[1]

        # Creating the FAISS-db index with IndexFlatL2 to store the embeddings in a vector-index - the simplest, most basic index, 
        # that calculates the exact distance (L2 distance) between our query and all vectors.
        index = faiss.IndexFlatL2(dims)

        # Adding our chunk embeddings to the index. We must convert to float32 for FAISS
        index.add(np.array(chunk_embeddings).astype('float32'))

        self.chunk_embeddings = np.array(chunk_embeddings).astype('float32')

        # print(f"FAISS index created with {index.ntotal} vectors.")
        return index, chunks

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
        self.query_embedding = self.embedding_model.encode([query]).astype('float32')

        # Search in the FAISS index for the top_k most similar chunks
        distances, indices = self.index.search(self.query_embedding, top_k)
        self.retrieved_indices = indices[0].tolist()

        # Retrieve the actual text chunks based on the indices
        retrieved_chunks = [self.chunks[i] for i in indices[0]]

        # Print the retrieved chunks and their distances
        # for i, chunk in enumerate(retrieved_chunks):
        #     print(f"Chunk {i+1} (Distance: {distances[0][i]}):\n{chunk}\n")
        return retrieved_chunks, distances[0]

    def get_embedding_visualization_data(self) -> tuple[List[str], np.ndarray, np.ndarray, List[int]]:
        """
        Returns everything needed to plot the embedding space: all chunk texts, all chunk embeddings, the query embedding,
        and the indices of the retrieved (top-k) chunks. Should be called after preprocess_document().
        """
        if self.chunk_embeddings is None or self.query_embedding is None:
            raise ValueError("No embeddings available. Call preprocess_document() first.")
        return self.chunks, self.chunk_embeddings, self.query_embedding, self.retrieved_indices

if __name__ == "__main__":
    config_data = load_config("src/rag/configs/rag_simple.yaml")
    PROMPT_TEMPLATE             = config_data['model'].get("llm_prompt", "You are a helpful assistant that answers questions based on the context provided. If you don't know the answer, just say 'I don't have that information'. Do not try to make up an answer, use only given information in this context :")
    MODELS_TO_TEST              = config_data['model'].get("instruct_completion_models", [])
    TEXT_SPLITTER               = config_data.get("text_splitter", [])
    SENTENCE_TRANSFORMER_MODEL  = config_data.get("sentence_transformer_model", "all-MiniLM-L6-v2")
    EMBEDDINGS_TOP_K            = config_data.get("embeddings_top_k", 3)
    LLM_MAX_TOKENS              = config_data['model'].get("llm_max_tokens", 200)

    configuration_data = {
        "llm_prompt"                    : PROMPT_TEMPLATE,
        "ts_chunk_size"                 : TEXT_SPLITTER.get("chunk_size", 150),
        "ts_chunk_overlap"              : TEXT_SPLITTER.get("chunk_overlap", 20),
        "sentence_transformer_model"    : SENTENCE_TRANSFORMER_MODEL,
        "embeddings_top_k"              : EMBEDDINGS_TOP_K,
        "llm_max_tokens"                : LLM_MAX_TOKENS
    }

    rag = SimpleRAG()
    #query = "What is the main topic of the document?"  
    query = 'What is the distance from Earth to the Sun?'

    #file_path = Path("data/raw/pdf/Full-47.pdf")
    file_path = Path("data/raw/txt/rag_notebook.txt")
    
    retrieved_chunks, distances = rag.preprocess_document(file_path=file_path, query=query, configuration_data=configuration_data)
    context = "\n\n".join(retrieved_chunks)

    # Checking for available models and selecting one for the LLM
    available_models = rag.get_working_models(MODELS_TO_TEST)
    model = random.choice(available_models)
    
    answer = rag.ask_model  (query          = query,
                                 prompt     = configuration_data["llm_prompt"], 
                                 context    = context, 
                                 model      = model,
                                 max_tokens = configuration_data["llm_max_tokens"])
    print(f"Answer: {answer}")
