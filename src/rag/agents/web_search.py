# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.config import load_config

# Data Science and NLP Libraries

# Standard Libraries
from ddgs import DDGS
import random

PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the user's question
                based *only* on the following search results. If the search results
                are empty or do not contain the answer, say 'I could not find
                any information on that.'"""

class WebSearchAgent(HuggingFaceClient):
    def __init__(self):
        super().__init__()
        # Geting a list of available models that work with the Hugging Face Inference API right now - they rotate availability on the free tier, so some may be down at any given time.
        data = load_config("src/rag/configs/rag_simple.yaml")
        MODELS_TO_TEST = data["model"].get("instruct_completion_models", [])
        print(f"Testing {len(MODELS_TO_TEST)} models for availability...")

        self.available_models = self.get_working_models(MODELS_TO_TEST)
        print(f"Available models: {self.available_models}")
        self.model = random.choice(self.available_models)

    def change_model(self, new_model: str):
        """
        Changes the model used for inference.

        Args:
            new_model (str): The name of the new model to use.
        """
        if new_model in self.available_models:
            self.model = new_model
            print(f"Model changed to {new_model}.")
        else:
            raise ValueError(f"{new_model} is not a working model. Please choose from: {self.available_models}")

    def web_search(self, query, max_results=5):
        """
        Performs a web search using DuckDuckGo and returns the results.
        Args:
            query (str): The search query.
            max_results (int): Maximum number of search results to return.
        Returns:
            str: Formatted string of search results or an error message.
        """
        try:
            results = DDGS().text(query, max_results=max_results)
            if not results:
                return "No search results found."
            
            # Debug - see what we're getting
            for r in results:
                print(f"  Found: {r['title']}")
            
            return "\n\n".join(
                f"Title: {r['title']}\n{r['body']}" for r in results
            )
        except Exception as e:
            return f"Search failed: {e}"


if __name__ == "__main__":
    agent = WebSearchAgent()

    query = "Who was on Apollo 11?"
    content = agent.web_search(query)

    answer = agent.ask_model(query = content, 
                           prompt = PROMPT_TEMPLATE,  
                           model = agent.model, 
                           context = query)
                           
    print(f"Answer: {answer}")

    

    