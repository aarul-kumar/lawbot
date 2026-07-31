# LawBot – RAG + LLM + AI Legal Aid

LawBot is a Retrieval-Augmented Generation (RAG) based AI Legal Assistant designed to provide context-grounded answers from uploaded legal PDF documents. The system ensures responses are strictly derived from document content, minimizing hallucinations and improving reliability for legal research and analysis.

The architecture supports both local and cloud-based large language models, making it flexible for development, experimentation, and deployment.

---

## Overview

LawBot enables users to:

- Upload legal documents (acts, case files, contracts, policies, etc.)
- Perform semantic search over document content
- Generate context-aware answers using a configurable LLM backend
- Operate fully offline using local models or leverage a cloud LLM via API

The system follows a modular RAG pipeline separating document processing, retrieval, and generation layers.

---

## System Architecture

<img width="1726" height="2543" alt="Flowchart" src="https://github.com/user-attachments/assets/557569e6-3006-40f5-b33b-e6f7789c85f1" />


---

## Core Features

- PDF ingestion and text extraction
- Recursive chunking with overlap for better semantic retrieval
- Embedding generation using sentence-transformers (local)
- FAISS-based vector similarity search
- Configurable LLM backend (Local or Groq API)
- Strict prompt design to prevent hallucinated responses
- Persistent local vector storage

---

## System Workflow

### Document Processing Phase
1. Upload PDF document.
2. Extract text using PDFPlumber.
3. Split text into overlapping chunks.
4. Generate embeddings using `nomic-embed-text`.
5. Store embeddings in a FAISS vector index.

### Query Processing Phase
1. User submits a question.
2. FAISS performs Top-K similarity retrieval.
3. Retrieved chunks are aggregated as context.
4. Context is injected into a structured prompt.
5. Selected LLM generates a grounded response.

---

## Technology Stack

| Layer | Technology |
|--------|------------|
| Frontend | Streamlit |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Local LLM | HuggingFace / transformers (configurable pipeline, default: distilgpt2) |
| Cloud LLM | Optional: configure Groq or other cloud LLMs in rag_pipeline.py |
| PDF Loader | PDFPlumber |

---

## Project Structure

```
LawBot/
│
├── frontend.py           # Streamlit application
├── rag_pipeline.py       # Retrieval and LLM integration
├── vector_database.py    # PDF processing and FAISS management
├── requirements.txt
├── pdfs/                 # Uploaded documents
└── vectorstore/          # Persisted FAISS index
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LawBot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

Create `requirements.txt` with the following (the repo already contains a `requirements.txt` example):

```
streamlit
python-dotenv
langchain
faiss-cpu
sentence-transformers
transformers
torch
pdfplumber
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Local LLM Setup

This project is configured to work fully locally using HuggingFace transformers pipelines. By default the code uses `distilgpt2` for the text-generation pipeline as a lightweight example. For better instruction-following models, set the environment variable `LOCAL_LLM` to a different model name (for example `bigscience/bloomz-1b1` or another model you have locally).

You do not need Ollama to run the default local flow.

### 5. Optional: Cloud LLM Setup (Groq or others)

If you want to use a cloud LLM backend instead of the local transformers pipeline, modify `rag_pipeline.py` to call your cloud LLM client. Example placeholder (update with your provider's SDK):

1. Create a `.env` file with provider credentials, e.g.:

```
GROQ_API_KEY=your_api_key_here
```

2. Edit `rag_pipeline.py` to initialize and call the cloud LLM client instead of the local pipeline. The code is deliberately simple so it is easy to swap the LLM backend.

---

## Running the Application

```bash
streamlit run frontend.py
```

Access the application at:

```
http://localhost:8501
```

---

## Reliability and Hallucination Control

The prompt design enforces:

- Strict use of provided context
- No fabricated legal information
- A fallback response:

```
"I don't know based on the provided document."
```

This ensures document-grounded, reliable outputs suitable for legal assistance workflows.
