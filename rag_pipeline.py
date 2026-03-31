from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from vector_database import load_vector_store

llm_model = None  # cache model


def get_llm():
    global llm_model

    if llm_model is None:
        # ✅ Use a lightweight HuggingFace model without importing transformers directly
        # This uses langchain-community to handle the pipeline internally
        llm_model = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",  # lightweight model
            task="text2text-generation",
            max_new_tokens=100,
            do_sample=False
        )

    return llm_model


def retrieve_docs(query, k=3):
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_context(documents):
    # ✅ limit context size to prevent memory issues
    return "\n\n".join([doc.page_content[:500] for doc in documents])


custom_prompt_template = """
You are a helpful legal assistant.

Answer the question ONLY using the context below.
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

    # ✅ clean output
    output = str(response)
    if "Answer:" in output:
        output = output.split("Answer:")[-1]

    return output.strip()[:300]  # limit output length