# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.clients.embedding_client import EmbeddingClient
from src.rag.config import load_config
from src.rag.documents.loaders import load_documents
from src.rag.documents.splitters import split_text_into_chunks
from src.rag.storage.chroma_store import ChromaStore

# Data Science Libraries
from langchain_chroma import Chroma

# Standard Libraries
import random


class VectorSearchRAG(HuggingFaceClient):
    def __init__(self, chroma_store: ChromaStore = None):
        # Initialize the InferenceClient for LLM
        super().__init__()

        self.vector_store = chroma_store  # Initialize vector_store as None
    

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

    # Initialize the EmbeddingClient and ChromaStore
    embedding_client = EmbeddingClient(model_name=SENTENCE_TRANSFORMER_MODEL)
    chroma_store = ChromaStore(embedding_client, persist_directory=config_data_vs["vector_db_path"])

    # Loading documents and splitting them into chunks
    documents = load_documents(folder_path=config_data_vs["document_folder_path"])
    all_chunks = []
    for doc_text in documents:
        _, chunks_doc = split_text_into_chunks(
            knowledge_text=doc_text,
            ts_chunk_size=TEXT_SPLITTER.get("chunk_size", 150),
            ts_chunk_overlap=TEXT_SPLITTER.get("chunk_overlap", 20)
        )
        all_chunks.extend(chunks_doc)

    query = "What is an Eclipse?"

    # Creating or getting the vector store via ChromaStore
    vector_store = chroma_store.get_or_create(chunks = all_chunks)

    # querying
    vector_search_rag = VectorSearchRAG(chroma_store=chroma_store)
    available_models = vector_search_rag.get_working_models(config_data_rag['model'].get("instruct_completion_models", []))
    model = random.choice(available_models)

    result = vector_search_rag.query_rag_system(
        query_text=query,
        vector_store=vector_store,
        prompt=config_data_rag['model'].get("llm_prompt"),
        model=model,
        top_k=config_data_rag.get("embeddings_top_k", 3)
    )
    print(f"Result: {result}")