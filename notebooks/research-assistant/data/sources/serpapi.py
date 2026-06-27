"""SerpAPI web-search connector.

Phase 1 Week 1 Day 3-4. Used as the agent's general-purpose web tool.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

logger = logging.getLogger(__name__)

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")


def fetch_serpapi(query: str, num_results: int = 5) -> list[dict[str, Any]]:
    """Call SerpAPI's Google endpoint and return normalized organic results.

    Each result has the shape ``{title, snippet, url, source: "web"}``.
    Returns an empty list when SerpAPI returns no organic results.
    """
    if not SERPAPI_API_KEY:
        raise RuntimeError(
            "SERPAPI_API_KEY is not set. Add it to .env before calling web_search."
        )

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    }
    raw = GoogleSearch(params).get_dict()

    if "error" in raw:
        raise RuntimeError(f"SerpAPI error: {raw['error']}")

    organic = raw.get("organic_results", []) or []
    normalized: list[dict[str, Any]] = []
    for item in organic[:num_results]:
        normalized.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": "web",
            }
        )
    logger.debug("SerpAPI returned %d results for %r", len(normalized), query)
    return normalized
