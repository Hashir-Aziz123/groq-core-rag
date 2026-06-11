from src.ingestion import PDFExtractor
from src.chunking import SlidingWindowChunker
from src.embeddings import LocalEmbedder

def test_full_pipeline():
    # 1. Extract
    print("\n--- PHASE 1: EXTRACTION ---")
    extractor = PDFExtractor("attention_is_all_you_need.pdf")
    pages = extractor.extract_clean_text()
    
    # 2. Chunk
    print("\n--- PHASE 2: CHUNKING ---")
    chunker = SlidingWindowChunker(chunk_size=1000, overlap=200)
    chunks = chunker.process_documents(pages)
    print(f"Generated {len(chunks)} overlapping chunks.")
    
    # 3. Embed
    print("\n--- PHASE 3: EMBEDDING ---")
    embedder = LocalEmbedder()
    embedded_chunks = embedder.embed_chunks(chunks)
    
    # Verification
    sample = embedded_chunks[0]
    vector = sample["embedding"]
    print("\n--- PIPELINE SUCCESS ---")
    print(f"Sample Chunk ID: {sample['chunk_id']}")
    print(f"Vector Dimensionality: {len(vector)}")
    print(f"Vector Preview: {vector[:5]} ...")

if __name__ == "__main__":
    test_full_pipeline()