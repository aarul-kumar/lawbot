import os
import streamlit as st
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

PDFS_DIRECTORY = "pdfs"
FAISS_DB_PATH = "vectorstore/db_faiss"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def upload_pdf(file):
    """Save uploaded file to PDFs directory and return saved path."""
    if not os.path.exists(PDFS_DIRECTORY):
        os.makedirs(PDFS_DIRECTORY)

    file_path = os.path.join(PDFS_DIRECTORY, file.name)

    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    return file_path


def load_pdf(file_path):
    """Load PDF and return a list of LangChain Document objects."""
    loader = PDFPlumberLoader(file_path)
    return loader.load()


def create_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into overlapping text chunks for semantic search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def get_embedding_model():
    """Return a LangChain embedding wrapper using sentence-transformers."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vector_store(text_chunks):
    """Create a FAISS vectorstore from document chunks and persist it locally."""
    if not os.path.exists(os.path.dirname(FAISS_DB_PATH)):
        os.makedirs(os.path.dirname(FAISS_DB_PATH), exist_ok=True)

    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local(FAISS_DB_PATH)

    return vector_store


@st.cache_resource
def load_vector_store():
    """Load the persisted FAISS vectorstore. Raises a clear error if missing."""
    if not os.path.exists(FAISS_DB_PATH):
        raise ValueError("⚠ No vector database found. Upload a PDF first or create a vectorstore.")

    embeddings = get_embedding_model()
    return FAISS.load_local(FAISS_DB_PATH, embeddings)
