import streamlit as st
from vector_database import (
    upload_pdf,
    load_pdf,
    create_chunks,
    create_vector_store
)
from rag_pipeline import answer_query, retrieve_docs

st.set_page_config(page_title="LawBot RAG", layout="wide")
st.title("⚖️ LawBot - AI Legal Assistant")

# ✅ Chat history memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF and creating embeddings..."):
        file_path = upload_pdf(uploaded_file)
        documents = load_pdf(file_path)
        chunks = create_chunks(documents)
        create_vector_store(chunks)
    st.success("✅ PDF processed successfully!")

# ✅ Display chat history
for chat in st.session_state.chat_history:
    st.chat_message("user").write(chat["user"])
    st.chat_message("assistant").write(chat["assistant"])

# ✅ ChatGPT-style input
user_query = st.chat_input("Ask anything from the uploaded PDF...")

if user_query:
    st.chat_message("user").write(user_query)

    with st.spinner("Thinking..."):
        retrieved_docs = retrieve_docs(user_query)

        response = answer_query(
            documents=retrieved_docs,
            query=user_query
        )

    st.chat_message("assistant").write(response)

    # ✅ Save chat
    st.session_state.chat_history.append({
        "user": user_query,
        "assistant": response
    })

# ✅ Clear chat button
if st.button("🗑 Clear Chat"):
    st.session_state.chat_history = []