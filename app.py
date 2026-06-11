import streamlit as st
from src.rag_engine import RAGEngine

# Page config for a clean, minimalist layout
st.set_page_config(page_title="ArXiv RAG Engine", page_icon= None, layout="centered")

@st.cache_resource
def load_engine():
    return RAGEngine()

st.title("Local RAG: Foundation Models")
st.markdown("Querying 10 foundational Machine Learning papers using a custom FAISS index and Groq-accelerated inference.")

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load RAG Engine. Error: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the papers (e.g., 'How does the Transformer handle sequence order?')..."):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching vector space and synthesizing..."):
            try:
                response = engine.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Inference Error: {e}")