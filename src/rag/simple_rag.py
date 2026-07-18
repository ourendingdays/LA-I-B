# Custom Modules
from src.rag.hugging_face_client import HuggingFaceClient
from src.rag.utils import load_config

# Data Science Libraries
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Standard Libraries
from pathlib import Path
from pypdf import PdfReader
import random
from typing import Dict, List

class SimpleRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Initialize the InferenceClient for LLM
        self.hf_client = HuggingFaceClient()


    def run(self, file_path: Path, query: str, configuration_data: Dict) -> tuple[str, List[float], List[str]]:
        """
        Main method to run the RAG pipeline: loads and splits document, creates embeddings and index, performs retrieval of the most relevant chunks.

        Args:
            file_path           (Path) : Path to the document file.
            query               (str)  : User query for retrieval.
            configuration_data  (dict) : Configuration data loaded from YAML file.

        Returns:
            tuple: Retrieved context (str) and distances (list of floats).
        """
        # Load and split the document into chunks
        chunks = self.load_and_split_document(file_path = file_path, 
                                              ts_chunk_size = configuration_data["ts_chunk_size"],
                                              ts_chunk_overlap = configuration_data["ts_chunk_overlap"])

        # Create embeddings for the chunks and build a FAISS index
        self.index, self.chunks = self.create_embeddings_and_index(chunks = chunks, model_name = configuration_data.get("sentence_transformer_model", "all-MiniLM-L6-v2"))

        # Perform retrieval based on the user query
        context, distances = self.retrieve(query, top_k = configuration_data.get("embeddings_top_k", 3))

        return context, distances

    def load_and_split_document(self, file_path: Path, ts_chunk_size: int = 150, ts_chunk_overlap: int = 20) -> List[str]:
        """
        Loads a document (.txt and .pdf formats) from the given file path and splits it into chunks.

        Args:
            file_path         (Path)          : Path to the document file.
            ts_chunk_size     (int, optional) : Size of each chunk. Defaults to 150.
            ts_chunk_overlap  (int, optional) : Overlap between chunks. Defaults to 20
        Returns:
            list: List of text chunks.
        """
        if file_path.suffix.lower() == ".txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_text = f.read()
        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(file_path)
            knowledge_text = ""
            for page in reader.pages:
                knowledge_text += page.extract_text()
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. Only .txt and .pdf are supported.")

        # Initializing the Text Splitter, which tries to split on paragraphs ("\n\n"), then newlines ("\n"), then spaces (" "), to keep semantically related text together as much as possible.

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ts_chunk_size,
            chunk_overlap=ts_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_text(knowledge_text)
        # print(f"Total number of chunks created: {len(chunks)}")    
        return chunks

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

        # print(f"FAISS index created with {index.ntotal} vectors.")
        return index, chunks

    def retrieve(self, query: str, top_k: int = 3) -> tuple[str, List[float]]:
        """
        Retrieves the top_k most relevant chunks based on the user query.

        Args:
            query (str): User query for retrieval.
            top_k (int): Number of top relevant chunks to retrieve. Default is 3.

        Returns:
            tuple: Retrieved context (str) and distances (list of floats).
        """
        # Create embedding for the user query
        query_embedding = self.embedding_model.encode([query]).astype('float32')

        # Search in the FAISS index for the top_k most similar chunks
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve the actual text chunks based on the indices
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        # Print the retrieved chunks and their distances
        # for i, chunk in enumerate(retrieved_chunks):
        #     print(f"Chunk {i+1} (Distance: {distances[0][i]}):\n{chunk}\n")
        return context, distances[0]

    def answer_question(self, query: str, prompt: str, context: str, model: str, max_tokens: int = 200) -> str:
        """
        Generates an answer using the LLM based on the provided context.

        Args:
            query       (str): User query for retrieval.
            prompt      (str): Prompt template for the LLM.
            context     (str): Retrieved context to use for generating the answer.
            model       (str): Name of the LLM model to use for generating the answer.
            max_tokens  (int): Maximum number of tokens for the generated answer. Default is 200.

        Returns:
            str: Generated answer from the LLM.
        """

        # Generate an answer using the LLM with the retrieved context
        result = self.hf_client.client.chat_completion(
            messages=[{
                "role": "system",
                "content": f"{prompt} : \n{context}"
            }, {
                "role": "user",
                "content": query
            }],
            model=model,
            max_tokens=max_tokens
        )

        # print(f"--- GENERATED ANSWER ---\n{result.choices[0].message.content}\n")
        content = result.choices[0].message.content
        if content is None:
            raise ValueError(f"Model '{model}' returned no content. Full response: {result}")

        return content.strip()
    

if __name__ == "__main__":
    config_data = load_config("src/rag/configs/rag_simple.yaml")
    PROMPT_TEMPLATE             = config_data.get("llm_prompt", "You are a helpful assistant that answers questions based on the context provided. If you don't know the answer, just say 'I don't have that information'. Do not try to make up an answer.")
    MODELS_TO_TEST              = config_data.get("llm_models_to_test", [])
    TEXT_SPLITTER               = config_data.get("text_splitter", [])
    SENTENCE_TRANSFORMER_MODEL  = config_data.get("sentence_transformer_model", "all-MiniLM-L6-v2")
    EMBEDDINGS_TOP_K            = config_data.get("embeddings_top_k", 3)
    LLM_MAX_TOKENS              = config_data.get("llm_max_tokens", 200)

    configuration_data = {
        "llm_prompt": PROMPT_TEMPLATE,
        "ts_chunk_size": TEXT_SPLITTER.get("chunk_size", 150),
        "ts_chunk_overlap": TEXT_SPLITTER.get("chunk_overlap", 20),
        "sentence_transformer_model": SENTENCE_TRANSFORMER_MODEL,
        "embeddings_top_k": EMBEDDINGS_TOP_K,
        "llm_max_tokens": LLM_MAX_TOKENS
    }

    rag = SimpleRAG()
    #query = "What is the main topic of the document?"  
    query = 'What is the distance from Earth to the Sun?'

    #file_path = Path("data/raw/pdf/Full-47.pdf")
    file_path = Path("data/raw/txt/rag_notebook.txt")
    
    context, distances = rag.run(file_path=file_path, query=query, configuration_data=configuration_data)

    # Checking for available models and selecting one for the LLM
    available_models = rag.hf_client.get_working_models(MODELS_TO_TEST)
    model = random.choice(available_models)
    
    answer = rag.answer_question(query=query, 
                                 prompt = configuration_data["llm_prompt"], 
                                 context=context, 
                                 model=model,
                                 max_tokens=configuration_data["llm_max_tokens"])
    print(f"Answer: {answer}")

