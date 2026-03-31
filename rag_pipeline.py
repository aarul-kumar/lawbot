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
            "text2text-generation",              # ✅ stable task
            model="google/flan-t5-base",         # ✅ best for Streamlit
            max_new_tokens=256,
            do_sample=False                      # ✅ prevents crash
        )

        llm_model = HuggingFacePipeline(pipeline=pipe)

    return llm_model


def retrieve_docs(query, k=3):
    from vector_database import load_vector_store  # lazy import
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_context(documents):
    # ✅ limit size to avoid memory crash
    return "\n\n".join([doc.page_content[:500] for doc in documents])


custom_prompt_template = """
You are a helpful legal assistant.

Use ONLY the information provided in the context below.
If the answer is not present, say:
"I don't know based on the provided document."

Question:
{question}

Context:
{context}

Answer:
"""


def answer_query(documents, query):
    model = get_llm()
    context = get_context(documents)

    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model

    response = chain.invoke({
        "question": query,
        "context": context
    })

    # ✅ clean output
    return str(response).strip()