import os
import time

import streamlit as st

from rag_pipeline import answer_query, retrieve_docs
from vector_database import (
    get_document_metadata,
    load_vector_store,
    process_uploaded_documents,
    upload_pdf,
)

st.set_page_config(page_title="LawBot Legal AI", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_uploaded_name" not in st.session_state:
    st.session_state.last_uploaded_name = None
if "language" not in st.session_state:
    st.session_state.language = "en"


CSS = """
<style>
:root {
  color-scheme: dark;
}
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, #07111f 0%, #11243d 45%, #0f1d2b 100%);
}
[data-testid="stSidebar"] {
  background: rgba(6, 14, 24, 0.95);
}
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}
div[data-testid="stMetric"] {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 0.5rem 0.7rem;
}
.stButton > button {
  border-radius: 999px;
  border: 1px solid #4f83ff;
  color: #f0f7ff;
  background: linear-gradient(90deg, #2b64df, #4f83ff);
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.04);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def stream_answer(container, text: str) -> None:
    for char_index in range(0, len(text), 24):
        container.markdown(text[char_index : char_index + 24])
        time.sleep(0.015)


def clear_chat() -> None:
    st.session_state.chat_history = []


LANGUAGE_LABELS = {"en": "English", "hi": "हिंदी"}

st.title("⚖️ LawBot — Indian Legal AI Assistant")
st.caption("Evidence-first legal research for Indian law, uploaded documents, and practical legal workflows.")

with st.sidebar:
    st.header("Navigation")
    st.radio("Mode", ["Assistant", "Knowledge", "Workflows"], key="nav_mode")
    st.divider()
    st.subheader("Language")
    st.selectbox("Response language", options=["en", "hi"], format_func=lambda value: LANGUAGE_LABELS[value], key="language")
    st.divider()
    st.subheader("Upload documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        if st.session_state.last_uploaded_name != uploaded_file.name:
            with st.spinner("Processing the document and building the index..."):
                try:
                    upload_pdf(uploaded_file)
                    process_uploaded_documents()
                    st.session_state.last_uploaded_name = uploaded_file.name
                    st.toast("Document processed successfully.")
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
    st.divider()
    st.subheader("Document index")
    metadata = get_document_metadata()
    if metadata:
        for entry in metadata[-3:]:
            st.write(f"• {entry['name']}")
    else:
        st.caption("No document metadata yet. Upload a PDF to begin.")


col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Documents", len(get_document_metadata()))
with col2:
    st.metric("Chat turns", len(st.session_state.chat_history))
with col3:
    st.metric("Mode", st.session_state.nav_mode)

if st.session_state.nav_mode == "Knowledge":
    st.subheader("Indian legal knowledge layer")
    st.write("The assistant can provide general educational context about constitutional rights, remedies, and the BNS/BNSS/BSA transition, while clearly separating this from evidence-backed findings from uploaded documents.")
    st.info("For high-risk, urgent, criminal, or court-related matters, consult a qualified advocate.")
elif st.session_state.nav_mode == "Workflows":
    st.subheader("Guided workflows")
    st.write("Use these templates as starting points for non-urgent legal drafting tasks only. Review the output before using it in formal proceedings.")
    workflow_options = [
        "Legal notice outline",
        "Grievance letter",
        "Complaint draft",
        "RTI request",
        "Application template",
    ]
    selected_workflow = st.selectbox("Choose a workflow", workflow_options)
    st.text_area("Draft output", value=f"{selected_workflow}: review carefully before use.", height=120)
else:
    st.subheader("Ask about the uploaded documents")
    suggested_questions = [
        "What rights are discussed in this document?",
        "Summarize the key points in plain language.",
        "What should I do next if I need to act on this document?",
    ]
    for question in suggested_questions:
        if st.button(question, key=f"suggest-{question}"):
            st.session_state.chat_history.append({"role": "user", "content": question})

    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area("Enter your question", height=120, placeholder="Ask about the uploaded legal document or general Indian legal concepts...")
        submitted = st.form_submit_button("Ask LawBot")

    if submitted and user_query.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        try:
            vector_store = load_vector_store()
            retrieved_docs = retrieve_docs(user_query, k=4)
            response = answer_query(documents=retrieved_docs, query=user_query, language=st.session_state.language, web_search_enabled=False)
        except Exception as exc:
            response = {
                "answer": f"I could not answer right now because the assistant hit an error: {exc}",
                "citations": [],
                "sources": [],
                "knowledge": None,
                "web_results": [],
                "grounded": False,
                "language": st.session_state.language,
            }

        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"], "citations": response["citations"], "sources": response["sources"]})
        st.rerun()

    if st.session_state.chat_history:
        st.divider()
        for entry in reversed(st.session_state.chat_history):
            if entry["role"] == "user":
                with st.chat_message("user"):
                    st.write(entry["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(entry["content"])
                    if entry.get("citations"):
                        with st.expander("Sources and citations"):
                            for citation in entry["citations"]:
                                source_path = citation.get("source_path")
                                if source_path and os.path.exists(source_path):
                                    st.markdown(f"- {citation['document_name']} · page {citation['page']} · chunk {citation['chunk']} · [Open PDF]({source_path})")
                                else:
                                    st.markdown(f"- {citation['document_name']} · page {citation['page']} · chunk {citation['chunk']}")
                    if entry.get("sources"):
                        st.caption("Evidence sources: " + ", ".join(entry["sources"]))
    else:
        st.info("Upload a PDF and ask a question to begin. The assistant will ground its answer in retrieved evidence and use the legal knowledge layer where relevant.")

    st.divider()
    if st.button("Clear chat history"):
        clear_chat()
        st.rerun()
