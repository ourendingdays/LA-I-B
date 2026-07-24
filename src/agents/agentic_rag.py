# Custom Modules
from src.rag.hugging_face_client import HuggingFaceClient
from src.rag.utils import load_config, load_documents, split_text_into_chunks

# Data Science Libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Standard Libraries
import random
from typing import Literal, Dict


ROUTING_PROMPT = """You are a routing agent. Decide whether answering this question requires searching a specific document collection, or whether it can be answered directly from general knowledge.

Reply with exactly one word: "search" or "direct".

- "search" if the question asks about specific content, data, or facts that would be found in an uploaded document (e.g. "summarize the document", "what does the PDF say about X", "according to the report...").
- "direct" if it's general knowledge, casual conversation, or math/reasoning unrelated to any specific document.

Question: {query}

Answer (search or direct):"""

ANSWER_WITH_CONTEXT_PROMPT = "Use this context to answer the question. If the answer is not in the context, say \"I don't have that information in the document.\""


class AgenticRAG(HuggingFaceClient):
    def __init__(self):
        super().__init__()
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None

    def build_vector_store(self, folder_path: str, ts_chunk_size: int = 150, ts_chunk_overlap: int = 20, collection_name: str = "agentic_rag_collection") -> Chroma:
        """
        Loads all documents from a folder, chunks them, and builds an in-memory Chroma store.

        Args:
            folder_path      (str) : Path to the folder containing .txt/.pdf documents.
            ts_chunk_size     (int) : Chunk size for splitting.
            ts_chunk_overlap  (int) : Chunk overlap for splitting.
            collection_name   (str) : Name of the Chroma collection.

        Returns:
            Chroma: The built vector store, also stored on self.vector_store.
        """
        documents = load_documents(folder_path=folder_path)  # List[str], one per file

        all_chunks_doc = []
        for doc_text in documents:
            _, chunks_doc = split_text_into_chunks(
                knowledge_text=doc_text,
                ts_chunk_size=ts_chunk_size,
                ts_chunk_overlap=ts_chunk_overlap
            )
            all_chunks_doc.extend(chunks_doc)

        self.vector_store = Chroma.from_documents(
            documents=all_chunks_doc,
            embedding=self.embedding_function,
            collection_name=collection_name
        )
        return self.vector_store

    def agent_controller(self, query: str, model: str) -> Literal["search", "direct"]:
        """
        Decides whether a query needs document retrieval or can be answered directly.

        Args:
            query (str) : The user's question.
            model (str) : The Hugging Face model to use for routing.

        Returns:
            str: "search" or "direct".
        """
        response = self.ask_model(
            prompt=ROUTING_PROMPT.format(query=query),
            query=query,
            context="",
            model=model,
            max_tokens=6
        )
        decision = response.strip().lower()
        return "search" if "search" in decision else "direct"

    def answer(self, query: str, model: str, top_k: int = 3, max_tokens: int = 300) -> Dict[str, str]:
        """
        Routes the query, then either answers directly or retrieves from self.vector_store + answers.

        Args:
            query      (str) : The user's question.
            model      (str) : Model to use for both routing and answering.
            top_k      (int) : Number of chunks to retrieve if searching.
            max_tokens (int) : Max tokens for the final answer.

        Returns:
            dict: {"answer": str, "action": "search"|"direct"}
        """
        action = self.agent_controller(query, model=model)

        if action == "search":
            if self.vector_store is None:
                raise ValueError("No vector store built. Call build_vector_store() first.")

            retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            answer_text = self.ask_model(
                prompt=ANSWER_WITH_CONTEXT_PROMPT,
                query=query,
                context=context,
                model=model,
                max_tokens=max_tokens
            )
        else:
            answer_text = self.ask_model(
                prompt="You are a helpful assistant.",
                query=query,
                context="",
                model=model,
                max_tokens=max_tokens
            )

        return {"answer": answer_text, "action": action}


if __name__ == "__main__":
    agent = AgenticRAG()

    config_data = load_config("src/rag/configs/rag_simple.yaml")
    MODELS_TO_TEST              = config_data['model'].get("instruct_completion_models", [])
    TEXT_SPLITTER               = config_data.get("text_splitter", {})
    EMBEDDINGS_TOP_K            = config_data.get("embeddings_top_k", 3)
    LLM_MAX_TOKENS              = config_data['model'].get("llm_max_tokens", 200)

    available_models = agent.get_working_models(MODELS_TO_TEST)
    model = random.choice(available_models)

    folder_path = "data/raw/pdf/"
    agent.build_vector_store(
        folder_path=folder_path,
        ts_chunk_size=TEXT_SPLITTER.get("chunk_size", 150),
        ts_chunk_overlap=TEXT_SPLITTER.get("chunk_overlap", 20)
    )

    query1 = "Give me a 5-point summary from the PDF document. What are the key takeaways?"
    query2 = "Why are space rockets painted in white?"

    for query in (query1, query2):
        result = agent.answer(query=query, model=model, top_k=EMBEDDINGS_TOP_K, max_tokens=LLM_MAX_TOKENS)
        print(f"Query: {query}")
        print(f"Action: {result['action']}")
        print(f"Answer: {result['answer']}\n{'-'*50}")