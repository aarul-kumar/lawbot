import re
from typing import Dict, List, Optional

LEGAL_KNOWLEDGE_BASE = [
    {
        "id": "constitution",
        "title": "Constitution of India",
        "summary": "The Constitution of India is the supreme law of the land. It lays down the structure of government, fundamental rights, fundamental duties, directive principles, and the remedies available in case of constitutional violations.",
        "tags": ["constitution", "fundamental rights", "duties", "directive principles", "remedies"],
        "source_type": "general_educational",
    },
    {
        "id": "fundamental-rights",
        "title": "Fundamental Rights",
        "summary": "Fundamental Rights are a core part of the Constitution and protect citizens against arbitrary state action. They include rights such as equality, freedom of speech and expression, life and personal liberty, religious freedom, and cultural and educational rights.",
        "tags": ["rights", "fundamental rights", "constitutional rights"],
        "source_type": "general_educational",
    },
    {
        "id": "constitutional-remedies",
        "title": "Constitutional Remedies",
        "summary": "Constitutional remedies are the mechanisms available to enforce fundamental rights. The most well-known remedy is writ jurisdiction under Articles 32 and 226 of the Constitution.",
        "tags": ["writ", "remedy", "article 32", "article 226"],
        "source_type": "general_educational",
    },
    {
        "id": "criminal-law-transition",
        "title": "Indian Criminal Law Transition",
        "summary": "India's criminal law framework has transitioned from the IPC, CrPC, and the Indian Evidence Act to the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA). The old laws are historical and transitional references unless a specific provision remains effective.",
        "tags": ["bns", "bnss", "bsa", "ipc", "crpc", "evidence act"],
        "source_type": "general_educational",
    },
    {
        "id": "grievance-procedure",
        "title": "Legal Grievance and Complaint Workflow",
        "summary": "Common legal workflows include drafting a grievance letter, a complaint, a legal notice outline, an RTI request, or an application. These drafts should be reviewed by a qualified advocate before being used in formal proceedings.",
        "tags": ["grievance", "complaint", "notice", "rti", "application"],
        "source_type": "general_educational",
    },
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def get_relevant_legal_knowledge(query: str) -> Optional[Dict[str, str]]:
    if not query:
        return None
    query_terms = set(_normalize(query).split())
    scored = []
    for item in LEGAL_KNOWLEDGE_BASE:
        score = 0
        for tag in item["tags"]:
            tag_terms = set(_normalize(tag).split())
            if tag_terms & query_terms:
                score += 2
        text_terms = set(_normalize(item["summary"]).split())
        if text_terms & query_terms:
            score += 1
        if score:
            scored.append((score, item))
    if not scored:
        return None
    scored.sort(key=lambda entry: entry[0], reverse=True)
    best_score, best_item = scored[0]
    if best_score < 1:
        return None
    return {
        "title": best_item["title"],
        "summary": best_item["summary"],
        "source_type": best_item["source_type"],
    }
