"""Semantic Scholar (S2) connector using the public REST API.

Phase 2 Week 4 Day 1-2. Free tier requires no key but accepts one via
``SEMANTIC_SCHOLAR_API_KEY`` for higher rate limits.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_s2(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """TODO: return papers with ``citationCount``, ``abstract``, ``year``."""
    raise NotImplementedError
