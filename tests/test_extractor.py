import os
from pathlib import Path
from src.config import RAW_DATA_DIR
from src.ingestion import PDFExtractor

def run_extraction_test():
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {RAW_DATA_DIR}")
        return

    print(f"Found {len(pdf_files)} papers for processing.\n")
    print(f"{'File Name':<30} | {'Pages':<6} | {'Total Chars':<12} | {'Status':<10}")
    print("-" * 68)

    for pdf_path in pdf_files:
        try:
            extractor = PDFExtractor(pdf_path.name)
            pages_data = extractor.extract_clean_text()
            
            total_pages = len(pages_data)
            total_chars = sum(len(page["text"]) for page in pages_data)
            
            status = "PASSED" if total_chars > 1000 else "EMPTY_WARN"
            print(f"{pdf_path.name:<30} | {total_pages:<6} | {total_chars:<12,} | {status:<10}")
            
        except Exception as e:
            print(f"{pdf_path.name:<30} | {'ERR':<6} | {'0':<12} | {str(e)[:15]}")

if __name__ == "__main__":
    run_extraction_test()