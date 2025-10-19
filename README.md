# Chat With PDF (RAG) 📄🤖

A full-featured Retrieval-Augmented Generation (RAG) application built with **Python**, **LangChain**, **FAISS**, **OpenAI**, and **Streamlit** that enables conversational interactions with multiple PDF documents simultaneously.

## Features
- **Multi-PDF Processing**: Extract and index content across multiple uploaded PDF files simultaneously.
- **Semantic Chunking**: Employs `RecursiveCharacterTextSplitter` with tuned overlap to retain semantic boundaries.
- **In-Memory Vector Search**: Indexes document chunks with high-performance similarity search using `FAISS` and `text-embedding-3-small`.
- **Conversational Memory**: Rephrases follow-up queries using chat history to ensure retrieval remains context-aware.
- **Source Transparency**: Provides expandable citations showing exact chunks, document origins, and page numbers.
- **Interactive UI**: Clean, responsive Streamlit dashboard with real-time indexing and chat streaming.

## System Architecture
1. **Document Loading**: `pypdf` extracts text and page metadata from uploaded PDFs.
2. **Chunking**: Text split into 1000-character segments with 200-character overlaps.
3. **Vector Indexing**: Embedded using OpenAI `text-embedding-3-small` and stored in FAISS.
4. **History-Aware Retrieval**: Reformulates conversational queries into standalone prompts.
5. **Generation**: `gpt-4o-mini` synthesizes answers strictly grounded in retrieved chunks.

## Project Structure
- `src/pdf_loader.py`: PDF text extraction and document parsing
- `src/vectorstore.py`: Recursive chunking and FAISS vector indexing
- `src/rag_chain.py`: History-aware retrieval and conversational RAG chain
- `app.py`: Streamlit user interface and session management
- `requirements.txt`: Project dependencies

## Tech Stack
- **Framework**: LangChain
- **LLM & Embeddings**: OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- **Vector DB**: FAISS (CPU)
- **UI**: Streamlit
- **Parsing**: PyPDF
