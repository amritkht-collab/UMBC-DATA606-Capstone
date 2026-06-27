"""PubMed connector using ``Bio.Entrez`` from Biopython.

Phase 2 Week 2 Day 1-2. Requires ``ENTREZ_EMAIL`` in the environment.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "")


def fetch_pubmed(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """TODO: search PubMed via ``Bio.Entrez`` and normalize results."""
    raise NotImplementedError
