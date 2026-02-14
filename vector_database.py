import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Directories
PDFS_DIRECTORY = "pdfs/"
FAISS_DB_PATH = "vectorstore/db_faiss"

# IMPORTANT: Use embedding model (not LLM)
EMBEDDING_MODEL_NAME = "nomic-embed-text"


def upload_pdf(file):
    if not os.path.exists(PDFS_DIRECTORY):
        os.makedirs(PDFS_DIRECTORY)

    file_path = os.path.join(PDFS_DIRECTORY, file.name)

    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    return file_path


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def get_embedding_model():
    return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)


def create_vector_store(text_chunks):
    if not os.path.exists("vectorstore"):
        os.makedirs("vectorstore")

    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    vector_store.save_local(FAISS_DB_PATH)

    return vector_store


def load_vector_store():
    embeddings = get_embedding_model()
    return FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
