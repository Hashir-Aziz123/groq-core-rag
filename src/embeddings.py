from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[INFO] Booting up local encoder: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        print(f"[INFO] Computing 384-dimensional vectors for {len(chunks)} chunks...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist() 
            
        return chunks