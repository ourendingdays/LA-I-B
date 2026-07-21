# Custom Modules
from src.rag.utils import load_config

# Data Science and NLP Libraries
from huggingface_hub import InferenceClient

# Standard Libraries
from dotenv import load_dotenv
import os
import random
import requests
from typing import List

class HuggingFaceClient:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Hugging Face token not found in environment variables.")
        
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        self.client = InferenceClient(token=HF_TOKEN)
    
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

    def ask_model(self, query: str, prompt: str, context: str, model: str, max_tokens: int = 200) -> str:
        """
        Sends a chat completion request to the Hugging Face Inference API and generates an answer using the LLM based on the provided context.

        Args:
            query       (str): User query for retrieval.
            prompt      (str): Prompt template for the LLM.
            context     (str): Retrieved context to use for generating the answer.
            model       (str): Name of the LLM model to use for generating the answer.
            max_tokens  (int): Maximum number of tokens for the generated answer. Default is 200.

        Returns:
            str: Generated answer from the LLM.
        """
        result = self.client.chat_completion(
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
        content = result.choices[0].message.content
        if content is None:
            raise ValueError(f"Model '{model}' returned no content. Full response: {result}")
        
        return content.strip()

    def request_model(self, api_url: str, input_text: str, result: str) -> str:
        response = requests.post(api_url, headers=self.headers, json={"inputs": input_text}).json()
        if result not in response[0]:
            raise ValueError(f"Result key '{result}' not found in response. Full response: {response}")
        return response[0][result]


if __name__ == "__main__":
    hf_client = HuggingFaceClient()

    # Get a list of available models that work with the Hugging Face Inference API right now - they rotate availability on the free tier, so some may be down at any given time.
    data = load_config("src/rag/configs/rag_simple.yaml")
    MODELS_TO_TEST = data.get("instruct_completion_models", [])

    available_models = hf_client.get_working_models(MODELS_TO_TEST)
    model = random.choice(available_models)

    print(f"Selected model: {model}")