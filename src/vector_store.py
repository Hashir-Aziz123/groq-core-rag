import faiss
import json
import numpy as np
from pathlib import Path
from src.config import ROOT_DIR

class FAISSStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index_path = ROOT_DIR / "data" / "index" / "faiss_store.index"
        self.metadata_path = ROOT_DIR / "data" / "index" / "chunk_metadata.json"
        
        # IndexFlatL2 performs exact nearest-neighbor search using Euclidean distance.
        # Since all-MiniLM-L6-v2 normalizes its vectors, L2 distance is functionally 
        # mathematically identical to Cosine Similarity here.
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def build_index(self, embedded_chunks: list[dict]):
        """Takes our pipeline output, extracts the math for FAISS, and saves the text."""
        if not embedded_chunks:
            print("[WARNING] No chunks provided to FAISS.")
            return

        print(f"\n[INFO] Loading {len(embedded_chunks)} vectors into FAISS...")
        
        # 1. Extract the vectors into a continuous 2D numpy array of type float32.
        # FAISS will segfault if you don't strictly enforce float32.
        vectors = np.array([chunk["embedding"] for chunk in embedded_chunks], dtype=np.float32)
        
        # 2. Add the math to FAISS
        self.index.add(vectors)
        
        # 3. Add the text/metadata to our ledger
        # We strip the raw embeddings out of the metadata to save disk space, 
        # since FAISS is already storing the math.
        for chunk in embedded_chunks:
            safe_chunk = {
                "source": chunk["source"],
                "page_id": chunk["page_id"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"]
            }
            self.metadata.append(safe_chunk)
            
        print(f"[SUCCESS] Index built. Total vectors: {self.index.ntotal}")

    def save_to_disk(self):
        """Persists the split-brain system to the hard drive."""
        print("[INFO] Flushing FAISS index and metadata to disk...")
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)
            
        print("[SUCCESS] Vector database safely stored.")