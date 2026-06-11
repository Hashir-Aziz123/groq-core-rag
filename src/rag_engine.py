from src.retriever import FAISSRetriever
from src.groq_client import GroqClient

class RAGEngine:
    def __init__(self):
        print("[INFO] Initializing RAG Pipeline...")
        self.retriever = FAISSRetriever()
        self.llm = GroqClient(model="llama-3.1-8b-instant")

    def ask(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve the raw chunks from FAISS
        print(f"\n[INFO] Searching vector space for: '{question}'")
        hits = self.retriever.search(question, top_k=top_k)
        
        if not hits:
            return "No relevant context found in the database."

        # 2. Construct the context payload
        context_blocks = []
        for hit in hits:
            context_blocks.append(
                f"--- Source: {hit['source']} (Page {hit['page']}) ---\n{hit['text']}"
            )
        
        context_str = "\n\n".join(context_blocks)

        # 3. The strict RAG Prompt
        prompt = f"""You are an expert machine learning researcher. Answer the user's question based strictly on the provided context. 

If the context does not contain the answer, explicitly state that you lack the context to answer. Do not hallucinate external knowledge. 
When answering, briefly cite the source document used.

CONTEXT:
{context_str}

QUESTION:
{question}
"""
        # 4. Generate the synthesis
        print("[INFO] Context retrieved. Synthesizing answer...\n")
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.generate_content(messages, temperature=0.1)
        
        return response

