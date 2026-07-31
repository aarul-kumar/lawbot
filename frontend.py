import streamlit as st
from vector_database import (
    upload_pdf,
    load_pdf,
    create_chunks,
    create_vector_store,
    load_vector_store,
)
from rag_pipeline import answer_query, retrieve_docs

st.set_page_config(page_title="LawBot RAG", layout="wide")
st.title("⚖️ LawBot - RAG + LLM + AI Legal Aid")

# -----------------------------
# PDF upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF and creating embeddings..."):
        file_path = upload_pdf(uploaded_file)
        documents = load_pdf(file_path)
        chunks = create_chunks(documents)
        create_vector_store(chunks)
    st.success("✅ PDF processed successfully!")

# -----------------------------
# User query input
# -----------------------------
user_query = st.text_area(
    "Enter your question:",
    height=150,
    placeholder="Ask anything from the uploaded PDF..."
)

ask_question = st.button("Ask LawBot")

# -----------------------------
# Handle question submission
# -----------------------------
if ask_question:
    if user_query.strip() == "":
        st.error("⚠ Please enter a question.")
    else:
        # Ensure a vectorstore exists (either just created or persisted on disk)
        try:
            vs = load_vector_store()
        except Exception as e:
            st.error("⚠ No vector database found. Upload a PDF first to create embeddings.")
            st.write(e)
        else:
            # Display only the user question
            try:
                st.chat_message("user").write(user_query)
            except Exception:
                # Older streamlit versions may not have chat components
                st.write("User:", user_query)

            with st.spinner("Thinking..."):
                # Retrieve relevant docs and generate response
                retrieved_docs = retrieve_docs(user_query)
                response = answer_query(
                    documents=retrieved_docs,
                    query=user_query
                )

            # Display only the assistant's answer
            try:
                st.chat_message("assistant").write(response)
            except Exception:
                st.write("LawBot:", response)
