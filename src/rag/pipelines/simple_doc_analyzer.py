# Custom Modules
from src.rag.clients.hugging_face_client import HuggingFaceClient
from src.rag.config import load_config
from src.rag.documents.loaders import load_document
from src.rag.documents.splitters import split_text_into_passages

# Standard Libraries
from pathlib import Path
from typing import List

class SimpleDocumentAnalyzer(HuggingFaceClient):
    def __init__(self):
        # Initialize the InferenceClient for LLM
        super().__init__()
    
    def summarize_document(self, input_text: str, model: str) -> str:
        """
        Summarizes the input text using the specified model. 
        Has two options for summarization: 
            - Option 1 uses the Hugging Face Inference API directly.
            -  Option 2 uses the InferenceClient's summarization method.
        
        Args:
            input_text (str): The text to be summarized.
            model (str): The model to use for summarization.
        Returns:
            str: The summarized text.
        """
        # Option 1
        # API_URL = f"https://router.huggingface.co/hf-inference/models/{model}"
        # result1 = self.request_model(api_url=API_URL, input_text=input_text, result="summary_text")

        # Option 2
        result2 = self.client.summarization(
            input_text,
            model=model
        )

        return result2.summary_text

    def generate_questions(self, passage : str, min_questions: int=3, model: str = "Qwen/Qwen2.5-7B-Instruct") -> List[str]:
        """
        Takes a text passage as input and produces a list of questions. 

        Args:
            passage         (str)  : Text passage to generate questions from.
            min_questions   (int)  : Minimum number of questions to generate. Defaults to 3.
            model           (str)  : Model to use for question generation. Defaults to "Qwen/Qwen2.5-7B-Instruct".
        Returns:
            list: List of generated questions.
        """
        response = self.ask_model(prompt = "Generate questions based on the following passage.",
                                context = "",
                                model = model,
                                max_tokens = 300,
                                query= f"Generate exactly {min_questions} questions about the following text. Return only the questions, one per line, numbered.\n\nText: {passage}")
    
        questions = [q.strip().lstrip('0123456789.-) ') for q in response.split('\n') if '?' in q]
        
        # if we didn't get enough, ask for more from a chunk
        if len(questions) < min_questions:
            passage_sentences = passage.split('. ')
            for i in range(0, len(passage_sentences), 2):
                if len(questions) >= min_questions:
                    break
                chunk = '. '.join(passage_sentences[i:i+2])
                result_fallback = self.ask_model(prompt = "Generate questions based on the following passage.",
                                        context = "",
                                        model = model,
                                        max_tokens = 300,
                                        query= f"Generate 1 question about: {chunk}")
          
                if '?' in result_fallback:
                    questions.append(result_fallback.lstrip('0123456789.-) '))
        
        return questions[:min_questions]
    
    def generate_passages_with_questions(self, input_text: str, model: str = "Qwen/Qwen2.5-7B-Instruct") -> tuple[list[str], list[str]]:
        """
        Splits the input text into passages, generates questions for each passage, and retrieves answers for each question using the specified model.
        Args:
            input_text (str): The text to be analyzed.
            model (str): The model to use for question answering. Defaults to "Qwen/Qwen2.5-7B-Instruct".

        Returns:
            tuple: A tuple containing a list of passages and a dictionary mapping passage indices to question-answer pairs.
        """
        passages = split_text_into_passages(input_text)

        question_answer_passage = {}
        for idx, passage in enumerate(passages):
            question_answer_pairs = {}
            questions = self.generate_questions(passage)
            for question in questions:
                answer = self.ask_model(query=question, 
                                        prompt=f"Answer the following question based on the provided passage. Give a short, direct answer.", 
                                        context=passage,
                                        model=model, 
                                        max_tokens=150)
                question_answer_pairs[question] = answer
            
            question_answer_passage[idx] = question_answer_pairs

        return passages, question_answer_passage



if __name__ == "__main__":
    doc_analyzer = SimpleDocumentAnalyzer()
    document_path = Path("data/raw/pdf/Full-48.pdf")
    query = "What is the main topic of the document?"

    knowledge_text  = load_document(file_path = document_path)
    
    config_data = load_config("src/rag/configs/rag_simple.yaml")    
    MODEL                       = config_data['model'].get("summarization_models")[0]
    summary = doc_analyzer.summarize_document(input_text=knowledge_text, model=MODEL)
    
    print(f"Summary : {summary}")

    passages, passage_qa = doc_analyzer.generate_passages_with_questions(input_text = knowledge_text)
    for idx, qa_pairs in passage_qa.items():
        print(f"Passage {idx}: {passages[idx]}...")
        for question, answer in qa_pairs.items():
            print(f"  Q: {question}\n  A: {answer}")
