"""ArXiv connector using the ``arxiv`` Python package.

Phase 2 Week 2 Day 1-2.
"""

from __future__ import annotations

from typing import Any


def fetch_arxiv(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Return normalized papers from arXiv search results."""
    try:
        import arxiv
    except ImportError as exc:
        raise RuntimeError("arxiv package is not installed. Install requirements first.") from exc

    if max_results <= 0:
        return []

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: list[dict[str, Any]] = []
    for result in search.results():
        papers.append(
            {
                "title": (result.title or "").strip(),
                "abstract": (result.summary or "").strip(),
                "url": getattr(result, "entry_id", "") or "",
                "authors": [a.name for a in getattr(result, "authors", [])],
                "year": int(result.published.year) if getattr(result, "published", None) else None,
                "source": "arxiv",
            }
        )
        if len(papers) >= max_results:
            break
    return papers
