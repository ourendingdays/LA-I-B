# Custom Modules
from src.rag.visualization.charts_and_plots import load_config

# Data Science and NLP Libraries
from huggingface_hub import InferenceClient

# Standard Libraries
from ddgs import DDGS
from dotenv import load_dotenv
import os
import random
from typing import List

PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the user's question
                based *only* on the following search results. If the search results
                are empty or do not contain the answer, say 'I could not find
                any information on that.'"""

class WebSearchAgent:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Hugging Face token not found in environment variables.")

        self.client = InferenceClient(token=HF_TOKEN)

        # Get a list of available models that work with the Hugging Face Inference API right now - they rotate availability on the free tier, so some may be down at any given time.
        data = load_config("src/rag/configs/rag_simple.yaml")
        MODELS_TO_TEST = data["model"].get("instruct_completion_models", [])

        self.available_models = self.get_working_models(MODELS_TO_TEST)
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


    def get_working_models(self, models: List[str]) -> List[str]:
        """
        Returns a list of models that work with the Hugging Face Inference API.
        Some models may go down on HF's free tier — they rotate availability.

        Args:
            models (list): List of model names to test.

        Returns:
            list: List of model names that work with the Hugging Face Inference API.
        """
        working_models = []
        for model_name in models:
            does_work = self.test_model(model_name)

            if does_work:
                working_models.append(model_name)
        return working_models

    def test_model(self, model_name: str) -> bool:
        """
        Tests if a model works with the Hugging Face Inference API.

        Args:
            model_name (str): Name of the model to test.

        Returns:
            bool: True if the model works, False otherwise.
        """
        try:
            result = self.client.chat_completion(
                messages=[{"role": "user", "content": "Say hello"}],
                model=model_name,
                max_tokens=10
            )
            # print(f"{model_name}: WORKS - {result.choices[0].message.content}")
            return True
        except Exception as e:
            # print(f"{model_name}: FAILED - {str(e)[:80]}")
            return False
        
    def ask_alm(self, prompt: str, max_tokens:int = 300):
        """
        Sends a prompt to the specified model on Hugging Face and returns the response.
        """
        # Send the prompt to the model and get the response
        result = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=max_tokens
        )
        return result.choices[0].message.content.strip()

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

    def build_prompt(self, question: str, context: str) -> str:
        """
        Builds a prompt for the LLM using the question and context.

        Args:
            question (str): The user's question.
            context (str): The context retrieved from the web search.   
        
        Returns:
            str: The formatted prompt to send to the LLM.
        """
        return f"{PROMPT_TEMPLATE}\n\nContext:\n{context}\n\nQuestion:\n{question}"

if __name__ == "__main__":
    agent = WebSearchAgent()
    query = "Who was on Apollo 11?"
    context = agent.web_search(query)
    prompt = agent.build_prompt(query, context)
    answer = agent.ask_alm(prompt)
    print(f"Answer: {answer}")

    

    