import fitz  
import re
from pathlib import Path
from src.config import RAW_DATA_DIR

class PDFExtractor:
    def __init__(self, file_name: str):
        self.file_path = RAW_DATA_DIR / file_name
        if not self.file_path.exists():
            raise FileNotFoundError(f"Target PDF not found at: {self.file_path}")

    def extract_clean_text(self) -> list[dict]:
        cleaned_pages = []
        
        with fitz.open(self.file_path) as doc:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks", sort=True)
                page_lines = []
                
                for b in blocks:
                    block_text = b[4].strip()
                    if "arXiv:" in block_text or block_text.isdigit():
                        continue
                    page_lines.append(block_text)
                    
                raw_text = "\n".join(page_lines)
                clean_text = self._clean_pipeline(raw_text)
                
                cleaned_pages.append({
                    "page_id": page_num + 1,
                    "source": self.file_path.name,
                    "text": clean_text
                })
                
        return cleaned_pages

    def _clean_pipeline(self, text: str) -> str:
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()