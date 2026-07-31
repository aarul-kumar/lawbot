import json
import os
import re
from typing import List

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

PDFS_DIRECTORY = "pdfs"
FAISS_DB_PATH = "vectorstore/db_faiss"
METADATA_PATH = "vectorstore/documents.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize_filename(filename: str) -> str:
    """Reject path traversal and other unsafe upload names before writing to disk."""
    if not filename or not isinstance(filename, str):
        raise ValueError("Invalid filename provided for upload.")

    if os.path.isabs(filename) or "/" in filename or "\\" in filename:
        raise ValueError("Upload filename must not contain path separators.")

    basename = os.path.basename(filename)
    if basename in {"", ".", ".."}:
        raise ValueError("Upload filename is invalid.")

    if basename != filename:
        raise ValueError("Upload filename must not contain a path.")

    if not re.fullmatch(r"[A-Za-z0-9._-]+", basename):
        raise ValueError("Upload filename contains unsupported characters.")

    return basename


def _save_document_manifest(documents: List[Document]) -> None:
    _ensure_directory(os.path.dirname(METADATA_PATH))
    metadata = []
    for document in documents:
        source_path = document.metadata.get("source_path") or document.metadata.get("source", "")
        if source_path:
            metadata.append(
                {
                    "name": os.path.basename(source_path),
                    "path": source_path,
                    "page": document.metadata.get("page", 1),
                    "chunk_id": document.metadata.get("chunk_id", 1),
                }
            )
    if metadata:
        with open(METADATA_PATH, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


def upload_pdf(file) -> str:
    """Save an uploaded PDF to the local pdfs directory and return the saved path."""
    _ensure_directory(PDFS_DIRECTORY)
    safe_name = _sanitize_filename(file.name)
    file_path = os.path.join(PDFS_DIRECTORY, safe_name)
    with open(file_path, "wb") as handle:
        handle.write(file.getbuffer())
    return file_path


def load_pdf(file_path: str) -> List[Document]:
    """Load a PDF into LangChain documents, with a best-effort OCR fallback for scanned files."""
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    if documents:
        enriched_documents = []
        for index, document in enumerate(documents):
            metadata = dict(document.metadata or {})
            metadata.update({
                "source_path": file_path,
                "source": os.path.basename(file_path),
                "page": metadata.get("page", index + 1),
            })
            enriched_documents.append(Document(page_content=document.page_content, metadata=metadata))
        return enriched_documents

    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return [
            Document(
                page_content=(
                    f"Unable to extract readable text from {os.path.basename(file_path)}. "
                    "The PDF may be scanned or encrypted."
                ),
                metadata={"source_path": file_path, "source": os.path.basename(file_path), "page": 1},
            )
        ]

    images = convert_from_path(file_path)
    text_parts = []
    for image_index, image in enumerate(images, start=1):
        text_parts.append(pytesseract.image_to_string(image))
    content = "\n\n".join(part.strip() for part in text_parts if part.strip())
    return [
        Document(
            page_content=content or "No readable text could be extracted from the uploaded PDF.",
            metadata={"source_path": file_path, "source": os.path.basename(file_path), "page": 1, "ocr": True},
        )
    ]


def create_chunks(documents: List[Document], chunk_size: int = 900, chunk_overlap: int = 150):
    """Split documents into overlapping text chunks with metadata for retrieval and citations."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata.setdefault("source_path", metadata.get("source_path") or "")
        metadata.setdefault("source", os.path.basename(metadata.get("source_path", "uploaded-document.pdf")))
        metadata.setdefault("page", 1)
        metadata["chunk_id"] = index + 1
        chunk.metadata = metadata
    return chunks


def get_embedding_model():
    """Return a LangChain embedding wrapper using sentence-transformers."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_store(text_chunks: List[Document]):
    """Create a FAISS vectorstore from document chunks and persist it locally."""
    if not text_chunks:
        raise ValueError("No text chunks available to build a vectorstore.")
    _ensure_directory(os.path.dirname(FAISS_DB_PATH))
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local(FAISS_DB_PATH)
    _save_document_manifest(text_chunks)
    return vector_store


def process_uploaded_documents() -> List[Document]:
    """Load every PDF in the pdfs directory, chunk it, and build a combined vectorstore."""
    pdf_files = sorted(
        [
            os.path.join(PDFS_DIRECTORY, filename)
            for filename in os.listdir(PDFS_DIRECTORY)
            if filename.lower().endswith(".pdf")
        ]
    ) if os.path.exists(PDFS_DIRECTORY) else []
    if not pdf_files:
        raise ValueError("No PDFs have been uploaded yet.")

    documents: List[Document] = []
    for pdf_path in pdf_files:
        documents.extend(load_pdf(pdf_path))

    chunks = create_chunks(documents)
    create_vector_store(chunks)
    return chunks


@st.cache_resource
def load_vector_store():
    """Load the persisted FAISS vectorstore. Raises a clear error if missing."""
    if not os.path.exists(FAISS_DB_PATH):
        raise ValueError("⚠ No vector database found. Upload a PDF first or create a vectorstore.")

    embeddings = get_embedding_model()
    return FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)


def get_document_metadata() -> List[dict]:
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)
