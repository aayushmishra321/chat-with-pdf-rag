import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.pdf_loader import extract_text_from_pdfs
from src.vectorstore import split_documents, create_vector_store
from src.rag_chain import build_rag_chain

load_dotenv()

st.set_page_config(
    page_title="Chat with PDF (RAG)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Conversational Multi-PDF Assistant")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Sidebar Configuration and Uploads
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", "")
    )
    
    st.markdown("---")
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Index Documents", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting text and building FAISS vector index..."):
                try:
                    docs = extract_text_from_pdfs(uploaded_files)
                    chunks = split_documents(docs)
                    vectorstore = create_vector_store(chunks, api_key)
                    st.session_state.rag_chain = build_rag_chain(vectorstore, api_key)
                    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s).")
                except Exception as e:
                    st.error(f"Error indexing documents: {str(e)}")

# Display Message History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handle Query Input
user_query = st.chat_input("Ask a question about the uploaded documents...")
if user_query:
    if not st.session_state.rag_chain:
        st.info("Please upload and index documents in the sidebar first.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                response = st.session_state.rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history[:-1]
                })
                answer = response["answer"]
                st.markdown(answer)

                if "context" in response and response["context"]:
                    with st.expander("📌 View Referenced Sources"):
                        for i, doc in enumerate(response["context"], start=1):
                            source_name = doc.metadata.get("source", "Unknown Document")
                            page_no = doc.metadata.get("page", "N/A")
                            st.markdown(f"**Source {i}:** `{source_name}` — Page {page_no}")
                            st.caption(doc.page_content[:300] + "...")

        st.session_state.chat_history.append(AIMessage(content=answer))
