import os
from pathlib import Path

# Resolve the project root directory dynamically
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define explicit data paths
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
INDEX_DIR = ROOT_DIR / "data" / "index"

# Ensure directories exist upon import
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Global Configuration Parameters
CHUNK_SIZE = 500  # Token count per chunk
OVERLAP = 50      # Overlap token count
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"