"""Semantic Scholar (S2) connector using the public REST API.

Phase 2 Week 4 Day 1-2. Free tier requires no key but accepts one via
``SEMANTIC_SCHOLAR_API_KEY`` for higher rate limits.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any

from dotenv import load_dotenv

load_dotenv()

S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_s2(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch and normalize paper metadata from Semantic Scholar."""
    if limit <= 0:
        return []

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,url,authors,citationCount",
    }
    url = f"{S2_SEARCH_URL}?{urlencode(params)}"
    headers = {"Accept": "application/json"}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    request = Request(url=url, headers=headers, method="GET")
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    items = payload.get("data", []) or []
    papers: list[dict[str, Any]] = []
    for item in items[:limit]:
        authors = item.get("authors", []) or []
        papers.append(
            {
                "title": str(item.get("title", "")).strip(),
                "abstract": str(item.get("abstract", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "authors": [str(a.get("name", "")).strip() for a in authors if a.get("name")],
                "year": item.get("year"),
                "citationCount": int(item.get("citationCount", 0) or 0),
                "source": "semantic_scholar",
            }
        )
    return papers
