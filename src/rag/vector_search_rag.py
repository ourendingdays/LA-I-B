# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.clients.embedding_client import EmbeddingClient
from src.rag.config import load_config
from src.rag.documents.loaders import load_documents
from src.rag.documents.splitters import split_text_into_chunks

# Data Science Libraries
from langchain_chroma import Chroma

# Standard Libraries
import os
from pathlib import Path
import random


class VectorSearchRAG(HuggingFaceClient):
    def __init__(self, embedding_client: EmbeddingClient = None):
        # Initialize the InferenceClient for LLM
        super().__init__()

        self.embedding_client = embedding_client or EmbeddingClient()

        self.vector_store = None  # Initialize vector_store as None
        
    def create_vector_store(self, chunks, vector_db_path: Path, collection_name: str) -> Chroma:
        """
        Creates a vector store from the provided text chunks using Chroma and HuggingFaceEmbeddings.

        Args:
            chunks (list): List of text chunks to be stored in the vector store.
            vector_db_path (Path): Path to the directory where the vector store will be persisted.
            collection_name (str): Name of the collection to create in the vector store.
        Returns:
            Chroma: Created vector store instance.
        """
        vector_store = Chroma.from_documents(
            documents           = chunks,
            embedding           = self.embedding_client.self.embedding_client.get_langchain_embeddings(),
            persist_directory   = str(vector_db_path), # saves the database in a folder called ./chroma_db. That way, you don’t have to rebuild the database every time you restart the app; it stays saved.
            collection_name     = collection_name
        )
        return vector_store
    
    def load_vector_store(self, vector_db_path: Path, collection_name: str) -> Chroma:
        """
        Loads an existing vector store from the specified directory using Chroma and HuggingFaceEmbeddings.
        
        Args:
            vector_db_path (Path): Path to the directory where the vector store is persisted.
            collection_name (str): Name of the collection to load from the vector store.
        Returns:
            Chroma: Loaded vector store instance.
        """
        self.vector_store = Chroma(
            persist_directory   = str(vector_db_path),
            embedding_function  = self.embedding_client.embedding_function,
            collection_name     = collection_name
        )
        return self.vector_store

    def initialize_vector_store(self, folder_document_path: Path,
                            ts_chunk_size: int = 150,
                            ts_chunk_overlap: int = 20, 
                            vector_db_path: Path = "data/train/rag/chroma_db_src", 
                            collection_name: str = "rag_collection") -> Chroma:
        """
        Initializes the vector store. If the vector store does not exist, it creates one from the documents in the specified folder. If it exists, it loads the existing vector store.
        
        Args:
            folder_document_path    (Path): Path to the folder containing document files.
            ts_chunk_size           (int) : Size of each text chunk. Default is 1000.
            ts_chunk_overlap        (int) : Overlap between text chunks. Default is 20.
            vector_db_path          (Path): Path to the directory where the vector store will be persisted. Default is "data/train/rag/chroma_db_src".
            collection_name         (str) : Name of the collection to create or load in the vector store.
        Returns:
            Chroma: Initialized or loaded vector store instance.
        """
        if not os.path.exists(vector_db_path):
            
            documents = load_documents(folder_path=folder_document_path)
            print(f"Loaded {len(documents)} documents from {folder_document_path}")
            all_chunks_doc = []
            for doc_text in documents:
                _, chunks_doc = split_text_into_chunks(
                    knowledge_text=doc_text,
                    ts_chunk_size=ts_chunk_size,
                    ts_chunk_overlap=ts_chunk_overlap
                )
                all_chunks_doc.extend(chunks_doc)

            self.vector_store = self.create_vector_store(all_chunks_doc, vector_db_path=vector_db_path, collection_name=collection_name)
        else:
            # Load the existing vector store
            self.vector_store = self.load_vector_store(vector_db_path=vector_db_path, collection_name=collection_name)

        return self.vector_store
    
    def get_vector_store(self):
        """
        Returns the vector store instance. If it hasn't been initialized yet, raises an error.

        Returns:
            Chroma: The vector store instance.
        """
        if self.vector_store is None:
            raise ValueError("Vector store has not been initialized. Please call 'initialize_vector_store' first.")
        return self.vector_store

    def query_rag_system(self, query_text: str, vector_store: Chroma, prompt: str, model: str, top_k: int=3, max_tokens: int=300) -> str:
        """
        Links the user, the database, and the LLM : looks at the users question and finds the top 3 most relevant chunks (k=3). 
        Then, it puts those chunks into a strict prompt.

        Args:
            query_text (str): The user's query.
            vector_store (Chroma): The vector store instance to query.
            prompt (str): Prompt template for the LLM.
            model (str): Name of the LLM model to use for generating the answer.
            top_k (int): Number of top relevant chunks to retrieve. Default is 3.
        Returns:
            str: Generated answer from the LLM based on the retrieved context.  
        """
        # Retrieve top 3 relevant chunks
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query_text)
        #print(f"Docs ; {docs}")
        context = "\n\n".join(doc.page_content for doc in docs)

        # Generate answer via HF Inference API
        result = self.ask_model (
            query=query_text,
            prompt=prompt,
            context=context,
            model=model,
            max_tokens=max_tokens)

        return result


if __name__ == "__main__":
    config_data_rag = load_config("src/rag/configs/rag_simple.yaml")
    PROMPT_TEMPLATE             = config_data_rag['model'].get("llm_prompt")
    MODELS_TO_TEST              = config_data_rag['model'].get("instruct_completion_models", [])
    TEXT_SPLITTER               = config_data_rag.get("text_splitter", {})
    SENTENCE_TRANSFORMER_MODEL  = config_data_rag.get("sentence_transformer_model", "all-MiniLM-L6-v2")
    EMBEDDINGS_TOP_K            = config_data_rag.get("embeddings_top_k", 3)
    LLM_MAX_TOKENS              = config_data_rag['model'].get("llm_max_tokens", 200)

    configuration_data_rag = {
        "llm_prompt"                    : PROMPT_TEMPLATE,
        "ts_chunk_size"                 : TEXT_SPLITTER.get("chunk_size", 150),
        "ts_chunk_overlap"              : TEXT_SPLITTER.get("chunk_overlap", 20),
        "sentence_transformer_model"    : SENTENCE_TRANSFORMER_MODEL,
        "embeddings_top_k"              : EMBEDDINGS_TOP_K,
        "llm_max_tokens"                : LLM_MAX_TOKENS
    }

    config_data_vs = load_config("src/rag/configs/vector_store.yaml")
    configuration_data_vs = {
        "document_folder_path"          : config_data_vs.get("document_folder_path"),
        "vector_db_path"                : config_data_vs.get("vector_db_path"),
        "collection_name"               : config_data_vs.get("collection_name")
    }

    query = "What is an Eclipse?"
    
    vector_search_rag = VectorSearchRAG()
    available_models = vector_search_rag.get_working_models(MODELS_TO_TEST)

    vector_store = vector_search_rag.initialize_vector_store(
        folder_document_path=configuration_data_vs["document_folder_path"],
        ts_chunk_size=configuration_data_rag["ts_chunk_size"],
        ts_chunk_overlap=configuration_data_rag["ts_chunk_overlap"],
        vector_db_path=configuration_data_vs["vector_db_path"],
        collection_name=configuration_data_vs["collection_name"],
    )

    result = vector_search_rag.query_rag_system(
        query_text=query,
        vector_store=vector_store,
        prompt=configuration_data_rag["llm_prompt"],
        model=random.choice(available_models),
        top_k=configuration_data_rag.get("embeddings_top_k", 3)
    )

    print(f"Query: {query}\n")

    print(f"Result: {result}\n")
