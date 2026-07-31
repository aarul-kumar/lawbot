from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline
import os
import logging
from vector_database import load_vector_store

logging.basicConfig(level=logging.INFO)

# Simple cached transformer pipeline (text-generation). Using a small default model for local runs.
_llm_pipeline = None


def get_llm(model_name: str = os.getenv("LOCAL_LLM", "distilgpt2")):
    """Return a transformers text-generation pipeline. Cached in module scope."""
    global _llm_pipeline
    if _llm_pipeline is None:
        device = 0 if ("CUDA_VISIBLE_DEVICES" in os.environ and os.environ.get("CUDA_VISIBLE_DEVICES") != "") else -1
        _llm_pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2
        )
    return _llm_pipeline


def retrieve_docs(query, k: int = 3):
    """Retrieve top-k relevant document chunks from the vector store."""
    vs = load_vector_store()
    return vs.similarity_search(query, k=k)


def get_context(documents):
    """Concatenate document chunks into a single context string."""
    return "\n\n".join([getattr(d, "page_content", str(d)) for d in documents])


CUSTOM_PROMPT_TEMPLATE = (
    "Answer the user's question using ONLY the information in the context below.\n"
    "If the answer is not present in the context, respond exactly: I don't know based on the provided document.\n\n"
    "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:" 
)


def answer_query(documents, query: str) -> str:
    """Build a strict prompt using retrieved documents and ask the local LLM pipeline for an answer."""
    model = get_llm()
    context = get_context(documents)

    prompt_text = CUSTOM_PROMPT_TEMPLATE.format(question=query, context=context)

    # Call the transformers pipeline directly
    out = model(prompt_text, max_new_tokens=256, do_sample=True, temperature=0.2)

    # transformers text-generation pipeline returns a list with 'generated_text'
    generated = out[0].get("generated_text", "") if isinstance(out, list) and len(out) > 0 else str(out)

    # Remove the prompt prefix if model echoes it
    if generated.startswith(prompt_text):
        result = generated[len(prompt_text):]
    else:
        # Fallback: try to strip the original prompt if present
        result = generated.replace(prompt_text, "")

    # Return a cleaned string
    return result.strip()
