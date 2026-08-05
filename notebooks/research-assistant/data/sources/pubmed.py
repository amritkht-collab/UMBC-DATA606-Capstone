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
    """Search PubMed via Bio.Entrez and return normalized paper metadata."""
    if not ENTREZ_EMAIL:
        raise RuntimeError("ENTREZ_EMAIL is not set. Add it to .env before calling PubMed.")
    if max_results <= 0:
        return []

    try:
        from Bio import Entrez, Medline
    except ImportError as exc:
        raise RuntimeError("biopython is not installed. Install requirements first.") from exc

    Entrez.email = ENTREZ_EMAIL

    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    search_record = Entrez.read(search_handle)
    search_handle.close()

    id_list = search_record.get("IdList", [])
    if not id_list:
        return []

    fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="medline", retmode="text")
    records = Medline.parse(fetch_handle)

    papers: list[dict[str, Any]] = []
    for rec in records:
        pmid = rec.get("PMID", "")
        year = None
        date_str = rec.get("DP", "")
        if date_str:
            for token in str(date_str).split():
                if token.isdigit() and len(token) == 4:
                    year = int(token)
                    break

        papers.append(
            {
                "title": str(rec.get("TI", "")).strip(),
                "abstract": str(rec.get("AB", "")).strip(),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "authors": [str(a) for a in rec.get("AU", [])],
                "year": year,
                "source": "pubmed",
            }
        )

    fetch_handle.close()
    return papers[:max_results]
