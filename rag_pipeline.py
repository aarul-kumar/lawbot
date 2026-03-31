from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

llm_model = None  # cache model


def get_llm():
    global llm_model

    if llm_model is None:
        pipe = pipeline(
            "text-generation",
            model="distilgpt2",
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        llm_model = HuggingFacePipeline(pipeline=pipe)

    return llm_model


def retrieve_docs(query, k=3):
    from vector_database import load_vector_store  # lazy import (prevents cloud issues)
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])


# ✅ format chat history
def format_chat_history(history):
    return "\n".join(
        [f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history]
    )


# ✅ improved prompt
custom_prompt_template = """
You are a helpful legal assistant.

Use ONLY the information provided in the context below.
If the answer is not present, say:
"I don't know based on the provided document."

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""


def answer_query(documents, query, chat_history=[]):
    model = get_llm()
    context = get_context(documents)
    history_text = format_chat_history(chat_history)

    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model

    response = chain.invoke({
        "question": query,
        "context": context,
        "history": history_text
    })

    # ✅ FIX: clean output (distilgpt2 returns full text)
    if isinstance(response, str):
        return response.split("Answer:")[-1].strip()
    else:
        return str(response)