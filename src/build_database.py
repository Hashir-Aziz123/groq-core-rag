import time
from pathlib import Path
from src.config import RAW_DATA_DIR
from src.ingestion import PDFExtractor
from src.chunking import SlidingWindowChunker
from src.embeddings import LocalEmbedder
from src.vector_store import FAISSStore

def build_full_database():
    start_time = time.time()
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("[ERROR] No PDFs found.")
        return

    all_pages = []
    
    # --- PHASE 1: EXTRACTION ---
    print(f"--- PHASE 1: Extracting {len(pdf_files)} papers ---")
    for pdf_path in pdf_files:
        print(f" -> Parsing {pdf_path.name}")
        extractor = PDFExtractor(pdf_path.name)
        all_pages.extend(extractor.extract_clean_text())

    # --- PHASE 2: CHUNKING ---
    print("\n--- PHASE 2: Chunking Text ---")
    chunker = SlidingWindowChunker(chunk_size=1000, overlap=200)
    all_chunks = chunker.process_documents(all_pages)
    print(f"Generated {len(all_chunks)} overlapping chunks.")

    # --- PHASE 3: EMBEDDING ---
    print("\n--- PHASE 3: Computing Vectors ---")
    # This is the compute-heavy step. For 10 papers, this might take 
    # a few minutes depending on your CPU.
    embedder = LocalEmbedder()
    embedded_chunks = embedder.embed_chunks(all_chunks)

    # --- PHASE 4: INDEXING ---
    print("\n--- PHASE 4: Building FAISS Database ---")
    db = FAISSStore()
    db.build_index(embedded_chunks)
    db.save_to_disk()

    elapsed = time.time() - start_time
    print(f"\n[PIPELINE COMPLETE] Time elapsed: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    build_full_database()
    