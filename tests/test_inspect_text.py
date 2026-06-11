import os
from pathlib import Path
from src.config import RAW_DATA_DIR, ROOT_DIR
from src.ingestion import PDFExtractor

def run_full_corpus_inspection():
    inspection_dir = ROOT_DIR / "data" / "inspection_dumps"
    inspection_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"[ERROR] No PDFs found in {RAW_DATA_DIR}")
        return

    print(f"Starting full text dump for {len(pdf_files)} papers...\n")

    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")
        try:
            extractor = PDFExtractor(pdf_path.name)
            pages = extractor.extract_clean_text()
            
            output_filename = pdf_path.stem + "_dump.txt"
            output_file = inspection_dir / output_filename
            
            with open(output_file, "w", encoding="utf-8") as f:
                for page in pages:
                    f.write(f"\n\n--- PAGE {page['page_id']} ---\n\n")
                    f.write(page["text"])
                    
            print(f" -> Saved to {output_file.relative_to(ROOT_DIR)}")
            
        except Exception as e:
            print(f" -> [ERROR] Failed to process {pdf_path.name}: {e}")

if __name__ == "__main__":
    run_full_corpus_inspection()