class SlidingWindowChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = self.chunk_size - self.overlap

    def chunk_text(self, text:str) -> list[str]:
        chunks = []
        text_length = len(text)

        if text_length <= self.chunk_size:
            return [text]
        
        for i in range(0, text_length, self.stride):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
            
            if i + self.chunk_size >= text_length:
                break
                
        return chunks
        
    
    def process_documents(self, ingested_data: list[dict]) -> list[dict]:
        chunked_data = []
        
        for doc in ingested_data:
            text = doc.get("text", "")
            if not text:
                continue
                
            chunks = self.chunk_text(text)
            
            for idx, chunk in enumerate(chunks):
                chunked_data.append({
                    "source": doc["source"],
                    "page_id": doc["page_id"],
                    "chunk_id": idx,
                    "text": chunk
                })
                
        return chunked_data