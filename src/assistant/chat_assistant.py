# Custom Modules
from src.rag.clients.embedding_client import EmbeddingClient
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.config import load_config
from src.rag.documents.loaders import langchain_file_loaders, langchain_web_loader

# Data Science Libraries
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Standard Libraries
from pathlib import Path

class ChatAssistant(HuggingFaceClient):
    def __init__(self, embedding_client: EmbeddingClient = None):
        super().__init__()
        
        config_data = load_config("src/rag/configs/rag_simple.yaml")
        self.prompt             = config_data['model'].get("llm_prompt")
        MODELS_TO_TEST          = config_data['model'].get("instruct_completion_models", [])

        # Hugging Face Inference API Client Initialization
        model = self.get_working_models(models = MODELS_TO_TEST)  # Default model to use for answering queries    
        self.model = model[0] if model else None

        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embedding_client = embedding_client or EmbeddingClient(model_name=model_name)

        self.huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )

    def process_source_data(self, data: list[Document]):
        chunks = self.text_splitter.split_documents(data)

        ids = [str(i) for i in range(0, len(chunks))]
        self.vectordb = Chroma.from_documents(chunks, self.embedding_client.get_langchain_embeddings(), ids=ids) # persistent_directory="./chroma_db", collection_name="rag_notebook")

    def answer_query(self, query: str):
        """
        Answers a query using the vector database.

        Args:
            query (str): The query to answer.
        Returns:
            str: The answer to the query.
        """
        docs            = self.vectordb.similarity_search(query, k=3)
        similar_docs    = " ".join([doc.page_content for doc in docs])

        answer = self.ask_model(
            prompt=self.prompt,
            model=self.model,
            query=query,
            context=similar_docs,
            max_tokens=300,
        )
        return answer

    
    def load_file(self, file):
        """
        Loads a document from a file using LangChain's loaders.

        Args:
            file (str): Path to the document file.
        Returns:
            Document: Extracted text from the document.
        """
        return langchain_file_loaders(file)

    def load_web(self, url):
        """
        Loads a document from the web using LangChain's loaders.

        Args:
            url (str): Web URL of the website.
        Returns:
            Document: Extracted text from the document.
        """
        return langchain_web_loader(url)


if __name__ == "__main__":
    assistant = ChatAssistant()
    file_path = Path("data/raw/pdf/Full-49.pdf")
    url = "https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps"

    # Load document from file
    document_from_file = assistant.load_file(file_path)
    assistant.process_source_data(document_from_file)
    query = "What is the main topic of the document?"
    answer = assistant.answer_query(query)
    print(f"Answer to the query '{query}': {answer}")

    # Load document from web
    document_from_web = assistant.load_web(url)
    assistant.process_source_data(document_from_web)
    query = "What is st.chat_message in Streamlit?"
    answer = assistant.answer_query(query)
    print(f"Answer to the query '{query}': {answer}")