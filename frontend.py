import streamlit as st
from vector_database import (
    upload_pdf,
    load_pdf,
    create_chunks,
    create_vector_store
)
from rag_pipeline import answer_query, retrieve_docs, llm_model

st.set_page_config(page_title="LawBot RAG", layout="wide")
st.title("LawBot - RAG + LLM + AI Legal Aid")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Process PDF only once
if uploaded_file:
    with st.spinner("Processing PDF and creating embeddings..."):
        file_path = upload_pdf(uploaded_file)
        documents = load_pdf(file_path)
        chunks = create_chunks(documents)
        create_vector_store(chunks)
    st.success("✅ PDF processed successfully!")

# User input
user_query = st.text_area(
    "Enter your question:",
    height=150,
    placeholder="Ask anything from the uploaded PDF..."
)

ask_question = st.button("Ask LawBot")

if ask_question:
    if uploaded_file and user_query.strip() != "":
        st.chat_message("user").write(user_query)

        with st.spinner("Thinking..."):
            retrieved_docs = retrieve_docs(user_query)
            response = answer_query(
                documents=retrieved_docs,
                model=llm_model,
                query=user_query
            )

        st.chat_message("assistant").write(response)

    else:
        st.error("⚠ Please upload a PDF and enter a question.")
