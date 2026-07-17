# Data Science Libraries
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.rag.hugging_face_client import MODELS_TO_TEST, HuggingFaceClient

# Standard Libraries
from pathlib import Path
import random
from typing import List


class SimpleRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Initialize the InferenceClient for LLM
        self.hf_client = HuggingFaceClient()

        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(model_name)


    def run(self, file_path: Path, query: str) -> tuple[str, List[float], List[str]]:
        """
        Main method to run the RAG pipeline: load and split document, create embeddings and index, and perform retrieval.

        Args:
            file_path (Path): Path to the document file.
            query (str): User query for retrieval.

        Returns:
            tuple: Retrieved context (str), distances (list of floats), and available models (list of str).
        """
        # Load and split the document into chunks
        chunks = self.load_and_split_document(file_path = file_path)

        # Create embeddings for the chunks and build a FAISS index
        self.index, self.chunks = self.create_embeddings_and_index(chunks = chunks)

        # Perform retrieval based on the user query
        context, distances = self.retrieve(query)

        # Checking for available models and selecting one for the LLM
        available_models = self.hf_client.get_working_models(MODELS_TO_TEST)

        return context, distances, available_models

    def load_and_split_document(self, file_path: Path):
        """
        Loads a document from the given file path and splits it into chunks.

        Args:
            file_path (Path): Path to the document file.
        Returns:
            list: List of text chunks.
        """

        # Loading the document
        with open(file_path, 'r', encoding='utf-8') as f:
            knowledge_text = f.read()
        
        # Initializing the Text Splitter, which tries to split on paragraphs ("\n\n"), then newlines ("\n"), then spaces (" "), to keep semantically related text together as much as possible.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,  # Max size of a chunk
            chunk_overlap=20, # Overlap to maintain context between chunks
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
            chunks (list): List of text chunks.
            model_name (str): Name of the embedding model to use. Default is "all-MiniLM-L6-v2" - small embedding model, that runs 100% even on not the most demanding local machines.
        Returns:
            tuple: FAISS index and the list of chunks.
        """
        
        embedding_model = SentenceTransformer(model_name)

        # Embedding all chunks. This will take a moment as the model "reads" and "understands" each chunk.
        chunk_embeddings = embedding_model.encode(chunks)

        # print(f"Shape of the embeddings: {chunk_embeddings.shape}")

        # Vector Store with FAISS-database to store the embeddings in a vector-index

        # dimension of our vectors are 384 (the size of the embedding model output)
        dims = chunk_embeddings.shape[1]

        # Creating the FAISS index with IndexFlatL2 - the simplest, most basic index, that calculates the exact distance (L2 distance) between our query and all vectors.
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

    def answer_question(self, query: str, context: str, model: str) -> str:
        """
        Generates an answer using the LLM based on the provided context.

        Args:
            query (str): User query for retrieval.
            context (str): Retrieved context to use for generating the answer.
            model (str): Name of the LLM model to use for generating the answer.

        Returns:
            str: Generated answer from the LLM.
        """

        # Generate an answer using the LLM with the retrieved context
        result = self.hf_client.client.chat_completion(
            messages=[{
                "role": "system",
                "content": f"You are a helpful assistant. Use the following context to answer the question:\n{context}"
            }, {
                "role": "user",
                "content": query
            }],
            model=model,  # Using a working model from Hugging Face
            max_tokens=200
        )

        # print(f"--- GENERATED ANSWER ---\n{result.choices[0].message.content}\n")
        content = result.choices[0].message.content
        if content is None:
            raise ValueError(f"Model '{model}' returned no content. Full response: {result}")

        return content.strip()
    

if __name__ == "__main__":
    rag = SimpleRAG()
    query = "What is the main topic of the document?" # What is the distance from Earth to the Sun?
    
    context, distances, available_models = rag.run(file_path=Path("data/raw/rag_notebook.txt"), query=query)
    model = random.choice(available_models)
    
    answer = rag.answer_question(query=query, context=context, model=model)
    print(f"Answer: {answer}")

