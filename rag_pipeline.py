import logging
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from transformers import pipeline

from legal_knowledge import get_relevant_legal_knowledge
from vector_database import load_vector_store
from web_search import search_web

load_dotenv()

logging.basicConfig(level=logging.INFO)

_llm_pipeline = None


def rewrite_query(query: str) -> str:
    """Create a slightly more retrieval-friendly reformulation of the user query."""
    cleaned = re.sub(r"\s+", " ", query.strip())
    if not cleaned:
        return cleaned
    lowered = cleaned.lower()
    if lowered.startswith("what is"):
        return f"{cleaned} legal meaning and context"
    if "right" in lowered or "rights" in lowered:
        return f"{cleaned} constitutional law and legal protections"
    if "complaint" in lowered or "grievance" in lowered or "notice" in lowered:
        return f"{cleaned} legal workflow drafting guidance"
    if "bns" in lowered or "bnss" in lowered or "bsa" in lowered:
        return f"{cleaned} current Indian criminal law framework"
    return cleaned


def _keyword_overlap_score(document_text: str, query: str) -> float:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    document_terms = set(re.findall(r"[a-z0-9]+", document_text.lower()))
    if not query_terms or not document_terms:
        return 0.0
    overlap = len(query_terms & document_terms)
    return overlap / max(1, len(query_terms))


def get_llm(model_name: Optional[str] = None):
    """Return a cached transformers text-generation pipeline."""
    global _llm_pipeline
    model_name = model_name or os.getenv("LOCAL_LLM", "distilgpt2")
    if _llm_pipeline is None:
        device = 0 if os.getenv("CUDA_VISIBLE_DEVICES") else -1
        _llm_pipeline = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.2,
        )
    return _llm_pipeline


def retrieve_docs(query: str, k: int = 4, source_filter: Optional[str] = None):
    """Retrieve top-k relevant document chunks using semantic similarity plus keyword overlap."""
    vector_store = load_vector_store()
    rewritten_query = rewrite_query(query)
    queries = [query, rewritten_query]

    scored_results = []
    for candidate_query in queries:
        try:
            semantic_results = vector_store.similarity_search_with_score(candidate_query, k=max(k * 2, 6))
        except Exception:
            semantic_results = []
        for document, semantic_score in semantic_results:
            text = getattr(document, "page_content", "") or ""
            keyword_score = _keyword_overlap_score(text, candidate_query)
            combined_score = (1.0 / (1.0 + float(semantic_score))) + keyword_score * 0.5
            if source_filter:
                source_name = (document.metadata or {}).get("source", "")
                if source_filter.lower() not in source_name.lower():
                    continue
            scored_results.append((document, combined_score))

    if not scored_results:
        return []

    unique_documents = []
    seen = set()
    for document, score in sorted(scored_results, key=lambda entry: entry[1], reverse=True):
        identity = (getattr(document, "page_content", "")[:200], document.metadata.get("source"), document.metadata.get("page"))
        if identity in seen:
            continue
        seen.add(identity)
        unique_documents.append(document)
    return unique_documents[:k]


def get_context(documents: List) -> str:
    """Concatenate document chunks into a single context string."""
    return "\n\n".join([getattr(document, "page_content", str(document)) for document in documents])


def build_citation(document) -> dict:
    metadata = getattr(document, "metadata", {}) or {}
    source_name = metadata.get("source") or "Uploaded document"
    page = metadata.get("page") or 1
    chunk_id = metadata.get("chunk_id") or 1
    source_path = metadata.get("source_path") or ""
    return {
        "document_name": source_name,
        "page": page,
        "chunk": chunk_id,
        "source_path": source_path,
    }


def build_prompt(question: str, context: str, knowledge: Optional[dict] = None, language: str = "en", web_results: Optional[list] = None) -> str:
    language_hint = "Answer in English." if language == "en" else "Answer in Hindi."
    knowledge_block = ""
    if knowledge:
        knowledge_block = (
            "\n\nRelevant legal knowledge note:\n"
            f"Title: {knowledge['title']}\n"
            f"Summary: {knowledge['summary']}\n"
            f"Source type: {knowledge['source_type']}"
        )

    web_block = ""
    if web_results:
        web_block = "\n\nOptional web findings:\n" + "\n".join(
            f"- {result['title']} ({result['source']}): {result['snippet']}"
            for result in web_results
        )

    return (
        "You are LawBot, an evidence-first legal assistant for Indian legal questions.\n"
        "Use only the retrieved document evidence and the legal knowledge note below.\n"
        "Do not invent legal provisions, case names, dates, citations, or facts.\n"
        "If the evidence is missing or weak, say so clearly and ask the user to upload a relevant document or refine the question.\n"
        "Separate the answer into three parts: Verified facts, Legal interpretation, and Guidance.\n"
        "Keep the answer concise and mention that it is educational information rather than legal advice.\n"
        f"{language_hint}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{context}\n"
        f"{knowledge_block}\n"
        f"{web_block}\n\n"
        "Answer:"
    )


def answer_query(documents, query: str, language: str = "en", web_search_enabled: bool = False) -> dict:
    """Build a grounded answer from retrieved evidence and a lightweight legal knowledge layer."""
    model = get_llm()
    context = get_context(documents)
    knowledge = get_relevant_legal_knowledge(query)
    web_results = []
    if web_search_enabled:
        web_results = search_web(query, max_results=2).get("results", [])

    prompt_text = build_prompt(query, context, knowledge=knowledge, language=language, web_results=web_results)
    result = model(prompt_text, max_new_tokens=220, do_sample=True, temperature=0.2)
    generated = result[0].get("generated_text", "") if isinstance(result, list) and result else str(result)
    answer_text = generated.replace(prompt_text, "", 1).strip()
    if not answer_text:
        answer_text = (
            "I could not generate a strong answer from the current evidence. Please upload a more relevant document or refine the question."
        )

    citations = [build_citation(document) for document in documents]
    return {
        "answer": answer_text,
        "citations": citations,
        "sources": [citation["document_name"] for citation in citations],
        "knowledge": knowledge,
        "web_results": web_results,
        "grounded": bool(documents),
        "language": language,
    }
