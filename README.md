# DocuMind

A production-grade Retrieval-Augmented Generation (RAG) application for conversational document analysis. Built with LangChain, FAISS, and Streamlit. Supports both OpenAI and Groq API keys with automatic provider detection.

---

## Overview

DocuMind indexes PDF documents into a local FAISS vector store using on-device sentence embeddings and enables multi-turn conversational querying with source attribution. It is designed to work with any standard OpenAI-compatible API key or Groq API key without configuration changes.

---

## Features

- Multi-PDF ingestion with page-level metadata extraction
- Local sentence embeddings via `all-MiniLM-L6-v2` — no embedding API costs
- FAISS in-memory vector index with similarity search
- History-aware query reformulation for accurate follow-up retrieval
- Grounded answer generation with source chunk citations
- Session-based authentication with email and API key validation
- Chunk distribution analytics: bar chart, scatter plot, and page-level heatmap
- Automatic provider routing between OpenAI and Groq backends
- Model selector for Groq with all available chat-capable models

---

## Architecture

```
PDF Upload
    |
    v
Text Extraction (pypdf, page-level)
    |
    v
Recursive Text Chunking (800 chars, 150 overlap)
    |
    v
Local Embedding (all-MiniLM-L6-v2 via sentence-transformers)
    |
    v
FAISS Vector Index
    |
    v
Query Input
    |
    v
History-Aware Reformulation (if chat history exists)
    |
    v
Similarity Search (top-5 chunks)
    |
    v
Answer Generation (Groq or OpenAI LLM)
    |
    v
Response + Source Citations
```

---

## Supported API Providers

| Provider | Key Format | Default Model |
|----------|-----------|---------------|
| Groq     | `gsk_...` | `qwen/qwen3.6-27b` |
| OpenAI   | `sk-...`  | `gpt-4o-mini` |

When using Groq, a model selector is displayed in the sidebar with all available chat-capable models.

---

## Project Structure

```
.
├── app.py                  Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py       PDF text extraction
│   ├── vectorstore.py      Document chunking and FAISS indexing
│   └── rag_chain.py        RAG chain with provider routing
├── requirements.txt        Python dependencies
├── .env.example            Environment variable template
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A Groq API key (`gsk_...`) or OpenAI API key (`sk-...`)

### Installation

```bash
git clone https://github.com/your-username/documind.git
cd documind

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables (Optional)

Copy `.env.example` to `.env` and fill in your key. The app also accepts the key directly through the sign-in form.

```bash
cp .env.example .env
```

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
OPENAI_API_KEY=sk-your_openai_api_key_here
```

### Run

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

---

## Usage

1. Open the application in your browser
2. Click **Launch Workspace** on the landing page
3. Enter your email and API key, then click **Sign In**
4. Upload one or more PDF files in the sidebar
5. Optionally select a model (Groq keys show a model selector)
6. Click **Index Documents** to extract and embed document content
7. Switch to the **Conversation** tab and ask questions
8. Expand **Source References** under any response to see attributed chunks
9. Switch to the **Chunk Analytics** tab to inspect index distribution

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI Framework | Streamlit |
| LLM Orchestration | LangChain |
| LLM Providers | OpenAI, Groq |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (CPU) |
| PDF Parsing | pypdf |
| Visualization | Altair |
| Data Processing | pandas |

---

## Development Notes

- Embeddings are computed locally on CPU and do not require any API quota
- The FAISS index is stored in memory per session and must be rebuilt on restart
- Chat history is maintained in Streamlit session state
- All session data is cleared on sign-out or page refresh

---

## License

MIT
