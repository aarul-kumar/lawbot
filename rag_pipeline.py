from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

llm_model = None  # cache model


def get_llm():
    global llm_model

    if llm_model is None:
        # ✅ FORCE SAFE MODEL
        pipe = pipeline(
            task="text2text-generation",   # ✅ explicit
            model="google/flan-t5-small",  # ✅ lighter = safer on Streamlit
            max_new_tokens=100,
            do_sample=False               # ❌ NO SAMPLING (fix crash)
        )

        llm_model = HuggingFacePipeline(pipeline=pipe)

    return llm_model


def retrieve_docs(query, k=3):
    from vector_database import load_vector_store
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_context(documents):
    # ✅ STRICT LIMIT (VERY IMPORTANT)
    return "\n\n".join([doc.page_content[:300] for doc in documents])


custom_prompt_template = """
Answer the question using ONLY the context below.

If the answer is not present, say:
"I don't know based on the provided document."

Give a short answer (1 sentence).

Context:
{context}

Question:
{question}

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

    # ✅ CLEAN OUTPUT
    output = str(response)

    if "Answer:" in output:
        output = output.split("Answer:")[-1]

    return output.strip()[:200]