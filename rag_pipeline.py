from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from vector_database import load_vector_store

llm_model = None  # cache model

# -----------------------------
# Get LLM (instruction-following)
# -----------------------------
def get_llm():
    global llm_model

    if llm_model is None:
        pipe = pipeline(
            "text-generation",
            model="distilgpt2",  # replace with a stronger instruction-following model if possible
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3
        )
        llm_model = HuggingFacePipeline(pipeline=pipe)

    return llm_model

# -----------------------------
# Retrieve most relevant documents
# -----------------------------
def retrieve_docs(query, k=1):  # Only top chunk to keep context short
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)

# -----------------------------
# Combine chunks into context string
# -----------------------------
def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# -----------------------------
# Prompt template
# -----------------------------
custom_prompt_template = """
Answer the user's question using ONLY the information in the context below.
If the answer is not present in the context, respond: "I don't know based on the provided document."

Question:
{question}

Context:
{context}

Answer:
"""

# -----------------------------
# Generate response
# -----------------------------
def answer_query(documents, query):
    model = get_llm()
    context = get_context(documents)

    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model

    response = chain.invoke({
        "question": query,
        "context": context
    })

    # Clean up any stray newlines
    return response.strip()