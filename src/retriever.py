import faiss
import json
import numpy as np
from pathlib import Path
from src.config import ROOT_DIR
from src.embeddings import LocalEmbedder

class FAISSRetriever:
    def __init__(self):
        self.index_dir = ROOT_DIR / "data" / "index"
        self.index_path = self.index_dir / "faiss_store.index"
        self.metadata_path = self.index_dir / "chunk_metadata.json"
        
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Database files missing. Did you run build_database.py?")
            
        print("[INFO] Mounting FAISS index into memory...")
        self.index = faiss.read_index(str(self.index_path))
        
        print("[INFO] Loading metadata ledger...")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        # We must use the exact same model that generated the database
        self.embedder = LocalEmbedder()

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Embeds the query, searches FAISS, and maps back to text."""
        # 1. Embed the query. LocalEmbedder returns a list of dicts, so we extract just the vector.
        # We wrap the query in a list because the embedder expects a batch.
        query_chunk = [{"text": query}]
        embedded_query = self.embedder.embed_chunks(query_chunk)[0]["embedding"]
        
        # FAISS strictly requires a 2D numpy array of float32
        query_vector = np.array([embedded_query], dtype=np.float32)
        
        # 2. Perform the mathematical search
        # D = distances (lower is closer/better for L2), I = indices (the integer IDs)
        distances, indices = self.index.search(query_vector, top_k)
        
        # 3. Reconstruct the human-readable results
        results = []
        for rank, db_id in enumerate(indices[0]):
            if db_id == -1:
                # FAISS returns -1 if it couldn't find enough vectors (e.g., asking for top 5 in a DB of 3)
                continue
                
            chunk_data = self.metadata[db_id]
            results.append({
                "rank": rank + 1,
                "distance": float(distances[0][rank]),
                "source": chunk_data["source"],
                "page": chunk_data["page_id"],
                "text": chunk_data["text"]
            })
            
        return results
