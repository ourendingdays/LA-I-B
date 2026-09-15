from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be None or empty")

        print(self.model.encode([text]))
        
        return self.model.encode([text])[0]

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    # print(f"Model name: {semantic_search.model.__class__.__name__}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")
    return semantic_search

def embed_text(text: str):

    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")