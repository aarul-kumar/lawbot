# LawBot – Indian Legal AI Assistant

LawBot is a locally runnable, evidence-first legal assistant designed for Indian legal research and legal-document understanding. It combines PDF ingestion, semantic retrieval, a lightweight legal knowledge layer, and an accessible Streamlit frontend to help users ask grounded questions about uploaded legal documents and general Indian legal topics.

The project is intentionally modular and portfolio-friendly: it preserves a simple local-first workflow while layering in modern UX, better retrieval behaviour, clearer citations, and responsible legal guidance.

## What the app does

- Uploads and processes legal PDFs locally
- Builds a FAISS vector index over PDF content
- Retrieves relevant document chunks with semantic + keyword overlap scoring
- Produces grounded answers with source/citation-style references
- Adds a lightweight Indian-law knowledge layer for constitutional concepts and criminal-law transition guidance
- Offers an optional web-search integration hook (when an API key is configured)
- Presents a more polished Streamlit experience with better navigation and chat history

## Architecture

- Frontend: Streamlit
- PDF ingestion: PDFPlumber with OCR fallback (best effort)
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- Vector store: FAISS
- LLM: HuggingFace transformers pipeline (local-first, configurable via `LOCAL_LLM`)
- Knowledge layer: curated legal knowledge module for Indian law concepts

## Project structure

```text
LawBot/
├── frontend.py            # Streamlit UI and navigation
├── rag_pipeline.py        # Retrieval, reranking, prompt assembly, and answers
├── vector_database.py     # PDF upload, text extraction, chunking, FAISS persistence
├── legal_knowledge.py     # Lightweight Indian legal knowledge layer
├── web_search.py          # Optional web-search integration hook
├── requirements.txt       # Python dependencies
├── pdfs/                  # Uploaded PDFs
├── vectorstore/           # FAISS index and document metadata
├── eval/                  # Evaluation dataset and runner
└── tests/                 # Unit tests
```

## Installation

1. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install Python dependencies

```bash
pip install -r requirements.txt
```

3. Copy the sample environment file and edit it if needed

```bash
copy .env.example .env
```

Optional values:

```text
LOCAL_LLM=distilgpt2
SERPER_API_KEY=
```

## Running the app

```bash
streamlit run frontend.py
```

Open http://localhost:8501 in your browser.

## Usage notes

- Upload a PDF first to build or refresh the vector index.
- Ask questions that relate to the uploaded document or general Indian legal topics.
- Answers are evidence-first and should be treated as educational legal information, not professional legal advice.
- For serious criminal, constitutional, financial, urgent, or court-related matters, consult a qualified legal professional.

## Testing

Run the unit tests:

```bash
pytest -q
```

Run the small evaluation harness:

```bash
eval\run_eval.py
```

## Limitations

- The default local model (`distilgpt2`) is intentionally lightweight for quick local use; stronger instruction-tuned models will improve answer quality.
- OCR fallback is best effort and depends on `pdf2image` and `pytesseract` being available.
- Web research is optional and only works when a supported API key is configured.
- The assistant should be used as a legal-learning aid, not a replacement for a qualified advocate.
