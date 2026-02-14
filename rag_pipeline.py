from dotenv import load_dotenv
load_dotenv()

#from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from vector_database import load_vector_store

# Groq LLM
#llm_model = ChatGroq(model="llama-3.3-70b-versatile")
llm_model = ChatOllama(model="deepseek-r1:1.5b")


def retrieve_docs(query, k=3):
    vector_store = load_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])


custom_prompt_template = """
You are a helpful assistant.

Use ONLY the information provided in the context below to answer the user's question.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Do NOT make up information.

Question:
{question}

Context:
{context}

Answer:
"""


def answer_query(documents, model, query):
    context = get_context(documents)

    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model

    response = chain.invoke({
        "question": query,
        "context": context
    })

    return response.content
