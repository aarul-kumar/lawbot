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
            "text2text-generation",              # ✅ correct task
            model="google/flan-t5-base",         # ✅ stable + good quality
            max_new_tokens=100,                  # ✅ limit output
            do_sample=False                      # ✅ no randomness
        )

        llm_model = HuggingFacePipeline(pipeline=pipe)

    return llm_model


def retrieve_docs(query, k=3):
    from vector_database import load_vector_store  # lazy import
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_context(documents):
    # ✅ limit context size (prevents crashes)
    return "\n\n".join([doc.page_content[:500] for doc in documents])


# ✅ IMPROVED PROMPT
custom_prompt_template = """
You are a helpful legal assistant.

Answer the question ONLY using the given context.
Give a short and precise answer (1-2 sentences).

If the answer is not in the context, say:
"I don't know based on the provided document."

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

    # ✅ CLEAN OUTPUT (VERY IMPORTANT)
    output = str(response)

    if "Answer:" in output:
        output = output.split("Answer:")[-1]

    output = output.strip().replace("\n", " ")

    return output[:300]  # limit length