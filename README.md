# Local RAG Pipeline

A high-performance, developer-focused Local Retrieval-Augmented Generation (RAG) pipeline built to index and query dense academic literature. Uses local embeddings via FAISS and offloads inference to Groq's hardware layer.

## Architecture
* **Chunker:** Sliding-window text extraction to preserve multi-page semantic context.
* **Embedder:** Local vector generation using `all-MiniLM-L6-v2`.
* **Retriever:** $k$-nearest neighbor dense search via a flattened FAISS index.
* **LLM Layer:** Deterministic inference via Groq API (`llama-3.1-8b-instant`, temperature = 0.1).

## Directory Structure
```text
.
├── data/
│   ├── raw/                 # Put source PDFs here
│   └── index/               # Serialized FAISS binaries & metadata
├── src/
│   ├── chunker.py           # Text extraction & parsing
│   ├── embedder.py          # Vector generation
│   ├── retriever.py         # FAISS search operations
│   ├── groq_client.py       # Groq API client
│   └── rag_engine.py        # Context assembly & prompt constraints
├── tests/
│   └── test_suite_groq.py   # Integrity & stress-test suite
├── app.py                   # Streamlit UI
├── requirements.txt
└── README.md

