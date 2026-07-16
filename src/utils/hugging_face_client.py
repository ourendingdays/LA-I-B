# Data Science and NLP Libraries
from huggingface_hub import InferenceClient

# Standard Libraries
from dotenv import load_dotenv
import os
import random
from typing import List

# How to pick a good model:
# You want models that are "Instruct" tuned (they follow instructions) and support chat_completion. Look for these patterns in the name: Instruct, it (Google's naming), Chat. Avoid base models (no instruction tuning — they just autocomplete text, not answer questions).
MODELS_TO_TEST = [
    # Qwen family
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    
    # Meta Llama family
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    
    # Google Gemma family
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    
    # Mistral family
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    
    # Microsoft Phi family
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-4-mini-instruct",
    
    # DeepSeek
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    
    # Others
    "HuggingFaceH4/zephyr-7b-beta",
    "NousResearch/Hermes-3-Llama-3.1-8B",
]




class HuggingFaceClient:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise ValueError("Hugging Face token not found in environment variables.")
        
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        self.client = InferenceClient(token=HF_TOKEN)

        # Get a list of available models that work with the Hugging Face Inference API right now - they rotate availability on the free tier, so some may be down at any given time.
        available_models = self.get_working_models(MODELS_TO_TEST)
        self.model = random.choice(available_models)

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


if __name__ == "__main__":
    hf_client = HuggingFaceClient()
    print(f"Selected model: {hf_client.model}")