from src.retriever import FAISSRetriever
from src.groq_client import GroqClient

class RAGEngine:
    def __init__(self):
        print("[INFO] Initializing RAG Pipeline...")
        self.retriever = FAISSRetriever()
        self.llm = GroqClient(model="llama-3.1-8b-instant")

    def ask(self, question: str, top_k: int = 6) -> str:
        print(f"\n[INFO] Searching vector space for: '{question}'")
        hits = self.retriever.search(question, top_k=top_k)
        
        if not hits:
            return "No relevant context found in the database."

        context_blocks = []
        for hit in hits:
            context_blocks.append(
                f"--- Source: {hit['source']} (Page {hit['page']}) ---\n{hit['text']}"
            )
        
        context_str = "\n\n".join(context_blocks)

        prompt = f"""You are an expert machine learning researcher. Answer the user's question based strictly on the provided context. 

CRITICAL CONSTRAINTS:
1. If the context does not contain the answer for any part of the question, explicitly state exactly what is missing.
2. Do not assume, extrapolate, or guess properties of one entity based on another. 
3. If an optimization technique, hyperparameter, or configuration is not explicitly stated in the context for an architecture, do not claim it is 'likely' or 'assumed' to be identical to another architecture.
4. Cite the source document and page number for every fact used.

CONTEXT:
{context_str}

QUESTION:
{question}
"""
        print("[INFO] Context retrieved. Synthesizing answer...\n")
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.generate_content(messages, temperature=0.1)
        
        return response

