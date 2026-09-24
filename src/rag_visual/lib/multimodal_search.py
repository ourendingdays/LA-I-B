from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str):
        image = Image.open(image_path)
        embedding = self.model.encode([image])[0]
        return embedding

    def encode_text(self, texts: list[str]):
        text_embeddings = self.model.encode(texts)
        return text_embeddings



    def search_by_image(self, image_path: str, documents: list[dict], limit: int = 5):
        image_embedding = self.embed_image(image_path)

        texts = [doc["title"] for doc in documents]
        text_embeddings = self.model.encode(texts)

        similarities = []
        for i, text_emb in enumerate(text_embeddings):
            score = np.dot(image_embedding, text_emb) / (np.linalg.norm(image_embedding) * np.linalg.norm(text_emb))
            similarities.append((float(score), documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:limit]


def verify_image_embedding(image_path: str):
    import json
    search = MultimodalSearch()
    embedding = search.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

    with open("data/rag_visual/movies.json", "r") as f:
        data = json.load(f)
    documents = data["movies"]

    results = search.search_by_image(image_path, documents)
    for title_score, doc in results:
        print(f"{doc['title']} (similarity: {title_score:.4f})")