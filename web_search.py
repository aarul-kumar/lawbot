import os
from typing import List, Dict, Any

import requests


def search_web(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Perform a best-effort web search using Serper (if configured)."""
    api_key = os.getenv("SERPER_API_KEY") or os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "enabled": False,
            "message": "No web-search API key configured. Web research is disabled.",
            "results": [],
        }

    try:
        endpoint = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": max_results}
        response = requests.post(endpoint, headers=headers, json=payload, timeout=12)
        response.raise_for_status()
        payload = response.json()
        results = []
        for item in payload.get("organic", [])[:max_results]:
            results.append(
                {
                    "title": item.get("title", "Untitled"),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "source": item.get("source", "Web"),
                }
            )
        return {"enabled": True, "message": "Web research completed.", "results": results}
    except Exception as exc:  # pragma: no cover - best effort path
        return {
            "enabled": False,
            "message": f"Web research was unavailable: {exc}",
            "results": [],
        }
